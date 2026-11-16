#ifdef BOREALIS_USE_VULKAN

#define VK_USE_PLATFORM_VI_NN 1

#include <algorithm>
#include <array>
#include <borealis/core/application.hpp>
#include <borealis/core/logger.hpp>
#include <borealis/core/util.hpp>
#include <borealis/core/vk_handle.hpp>
#include <borealis/platforms/switch/switch_video_vk.hpp>
#include <iterator>
#include <limits>
#include <nanovg/vk_renderer.hpp>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

namespace brls
{

struct SwitchVulkanVideoContext::State
{
    using Instance    = vk::RootHandle<VkInstance, vkDestroyInstance>;
    using Device      = vk::RootHandle<VkDevice, vkDestroyDevice>;
    using Surface     = vk::ChildHandle<VkInstance, VkSurfaceKHR, vkDestroySurfaceKHR>;
    using Swapchain   = vk::ChildHandle<VkDevice, VkSwapchainKHR, vkDestroySwapchainKHR>;
    using Image       = vk::ChildHandle<VkDevice, VkImage, vkDestroyImage>;
    using ImageView   = vk::ChildHandle<VkDevice, VkImageView, vkDestroyImageView>;
    using Memory      = vk::ChildHandle<VkDevice, VkDeviceMemory, vkFreeMemory>;
    using Semaphore   = vk::ChildHandle<VkDevice, VkSemaphore, vkDestroySemaphore>;
    using Fence       = vk::ChildHandle<VkDevice, VkFence, vkDestroyFence>;
    using CommandPool = vk::ChildHandle<VkDevice, VkCommandPool, vkDestroyCommandPool>;

    class DisplayResolutionEvent
    {
      public:
        DisplayResolutionEvent()
        {
            Result rc = appletGetDefaultDisplayResolutionChangeEvent(&event);
            ready     = R_SUCCEEDED(rc);
            if (!ready)
            {
                Logger::warning("switch: failed to get default display resolution change event ({:#x}), falling back to operation mode change event", rc);
                Logger::warning("switch: resolution might not change properly when docking while the app is running using 3rd party docks!");
            }
        }
        ~DisplayResolutionEvent()
        {
            if (ready)
                eventClose(&event);
        }
        DisplayResolutionEvent(const DisplayResolutionEvent&)            = delete;
        DisplayResolutionEvent& operator=(const DisplayResolutionEvent&) = delete;

        bool isReady() const { return ready; }

        bool poll()
        {
            if (!ready || R_FAILED(eventWait(&event, 0)))
                return false;
            eventClear(&event);
            return true;
        }

      private:
        ::Event event { };
        bool ready = false;
    };

    struct SwapchainImage
    {
        VkImage image;
        ImageView view;
        Semaphore renderFinished;
    };

    struct Frame
    {
        Fence fence;
        Semaphore imageAcquired;
        CommandPool commandPool;
        VkCommandBuffer upload = VK_NULL_HANDLE;
        VkCommandBuffer draw   = VK_NULL_HANDLE;
        std::vector<std::function<void()>> releases;
    };

    static constexpr VkExtent2D FallbackExtent { 1920, 1080 };

    DisplayResolutionEvent resolutionEvent;
    Instance instance;
    Surface surface;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memoryProperties { };
    uint32_t queueFamilyIndex = 0;
    std::vector<std::string> enabledExtensions;
    Device device;
    VkQueue queue = VK_NULL_HANDLE;
    std::mutex queueMutex;
    std::mutex frameMutex;
    std::mutex releaseMutex;

    VkFormat stencilFormat           = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags stencilAspect = 0;
    VkPresentModeKHR presentMode     = VK_PRESENT_MODE_FIFO_KHR;
    VkPresentModeKHR requestedMode   = VK_PRESENT_MODE_FIFO_KHR;
    VkExtent2D desiredExtent { };

    Swapchain swapchain;
    VkSurfaceFormatKHR surfaceFormat { };
    VkExtent2D extent { };
    std::vector<SwapchainImage> images;
    Memory stencilMemory;
    Image stencilImage;
    ImageView stencilView;
    VkExtent2D stencilExtent { };

    std::array<Frame, FramesInFlight> frames;
    uint32_t slot = FramesInFlight - 1;
    std::optional<uint32_t> imageIndex;
    bool rendering        = false;
    bool resetRequested   = false;
    bool recreateRequired = false;

    std::unique_ptr<nvg::VkRenderer> renderer;
    NVGcontext* nvgContext = nullptr;

    State()
    {
        createInstance();
        selectPhysicalDevice();
        createDevice();
        stencilFormat = chooseStencilFormat();
        stencilAspect = stencilFormat == VK_FORMAT_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        desiredExtent = queryDisplayExtent();
        presentMode = requestedMode = choosePresentMode(VideoContext::swapInterval);
        createSwapchain();
        createFrames();

        renderer = std::make_unique<nvg::VkRenderer>(
            nvg::VkRenderer::Device { physicalDevice, device.get(), memoryProperties, FramesInFlight },
            nvg::VkRenderer::Target { surfaceFormat.format, stencilFormat });
        nvgContext = renderer->createContext(NVG_ANTIALIAS | NVG_STENCIL_STROKES);
        if (!nvgContext)
            fatal("switch: failed to create the nanovg context");
    }

    ~State()
    {
        waitIdle();
        if (nvgContext)
            nvg::VkRenderer::deleteContext(nvgContext);
        renderer.reset();
        for (Frame& frame : frames)
            runReleases(frame);
    }

    State(const State&)            = delete;
    State& operator=(const State&) = delete;

    static void check(VkResult result, std::string_view what)
    {
        if (result != VK_SUCCESS)
            fatal(fmt::format("switch: {} failed ({})", what, static_cast<int>(result)));
    }

    template <typename T, typename Enumerate>
    static std::vector<T> enumerate(Enumerate&& call)
    {
        uint32_t count = 0;
        call(&count, nullptr);
        std::vector<T> items(count);
        call(&count, items.data());
        items.resize(count);
        return items;
    }

    void createInstance()
    {
        constexpr std::array extensions { VK_KHR_SURFACE_EXTENSION_NAME, VK_NN_VI_SURFACE_EXTENSION_NAME };

        const VkApplicationInfo app {
            .sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "borealis",
            .pEngineName      = "borealis",
            .apiVersion       = VK_API_VERSION_1_3,
        };
        const VkInstanceCreateInfo info {
            .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo        = &app,
            .enabledExtensionCount   = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        };
        VkInstance created = VK_NULL_HANDLE;
        check(vkCreateInstance(&info, nullptr, &created), "vkCreateInstance");
        instance = Instance(created);

        const VkViSurfaceCreateInfoNN surfaceInfo {
            .sType  = VK_STRUCTURE_TYPE_VI_SURFACE_CREATE_INFO_NN,
            .window = nwindowGetDefault(),
        };
        VkSurfaceKHR createdSurface = VK_NULL_HANDLE;
        check(vkCreateViSurfaceNN(instance.get(), &surfaceInfo, nullptr, &createdSurface), "vkCreateViSurfaceNN");
        surface = Surface(instance.get(), createdSurface);
    }

    std::optional<uint32_t> presentableGraphicsFamily(VkPhysicalDevice candidate) const
    {
        auto families = enumerate<VkQueueFamilyProperties>([&](uint32_t* count, VkQueueFamilyProperties* items)
            { vkGetPhysicalDeviceQueueFamilyProperties(candidate, count, items); });

        for (uint32_t index : std::views::iota(0u, static_cast<uint32_t>(families.size())))
        {
            VkBool32 presentable = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(candidate, index, surface.get(), &presentable);
            if ((families[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) && presentable)
                return index;
        }
        return std::nullopt;
    }

    void selectPhysicalDevice()
    {
        auto devices = enumerate<VkPhysicalDevice>([&](uint32_t* count, VkPhysicalDevice* items)
            { vkEnumeratePhysicalDevices(instance.get(), count, items); });

        for (VkPhysicalDevice candidate : devices)
        {
            VkPhysicalDeviceProperties properties;
            vkGetPhysicalDeviceProperties(candidate, &properties);
            if (properties.apiVersion < VK_API_VERSION_1_3)
                continue;

            if (auto family = presentableGraphicsFamily(candidate))
            {
                physicalDevice   = candidate;
                queueFamilyIndex = *family;
                vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
                Logger::info("switch: using {} for vulkan rendering", properties.deviceName);
                return;
            }
        }
        fatal("switch: no vulkan 1.3 device with a presentable graphics queue");
    }

    void createDevice()
    {
        auto available = enumerate<VkExtensionProperties>([&](uint32_t* count, VkExtensionProperties* items)
            { vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, count, items); });
        auto supported = [&](std::string_view name)
        {
            return std::ranges::any_of(available, [&](const VkExtensionProperties& extension)
                { return name == extension.extensionName; });
        };

        if (!supported(VK_KHR_SWAPCHAIN_EXTENSION_NAME))
            fatal("switch: the vulkan device does not support VK_KHR_swapchain");
        enabledExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

        for (std::string_view extension : { std::string_view(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME), std::string_view(VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME) })
            if (supported(extension))
                enabledExtensions.emplace_back(extension);

        VkPhysicalDeviceVulkan13Features supported13 { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
        VkPhysicalDeviceFeatures2 supportedFeatures { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &supported13 };
        vkGetPhysicalDeviceFeatures2(physicalDevice, &supportedFeatures);
        if (!supported13.dynamicRendering || !supported13.synchronization2)
            fatal("switch: the vulkan device lacks dynamic rendering or synchronization2");

        VkPhysicalDeviceVulkan13Features features13 {
            .sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .synchronization2 = VK_TRUE,
            .dynamicRendering = VK_TRUE,
        };

        std::vector<const char*> names;
        std::ranges::transform(enabledExtensions, std::back_inserter(names), [](const std::string& name)
            { return name.c_str(); });

        const float priority = 1.0f;
        const VkDeviceQueueCreateInfo queueInfo {
            .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = queueFamilyIndex,
            .queueCount       = 1,
            .pQueuePriorities = &priority,
        };
        const VkDeviceCreateInfo info {
            .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext                   = &features13,
            .queueCreateInfoCount    = 1,
            .pQueueCreateInfos       = &queueInfo,
            .enabledExtensionCount   = static_cast<uint32_t>(names.size()),
            .ppEnabledExtensionNames = names.data(),
        };
        VkDevice created = VK_NULL_HANDLE;
        check(vkCreateDevice(physicalDevice, &info, nullptr, &created), "vkCreateDevice");
        device = Device(created);
        vkGetDeviceQueue(device.get(), queueFamilyIndex, 0, &queue);

        for (const std::string& name : enabledExtensions)
            Logger::info("switch: enabled vulkan device extension {}", name);
    }

    VkFormat chooseStencilFormat() const
    {
        constexpr std::array candidates { VK_FORMAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT };
        auto usable = std::ranges::find_if(candidates, [&](VkFormat format)
            {
                VkFormatProperties properties;
                vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
                return (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0; });
        if (usable == candidates.end())
            fatal("switch: no stencil attachment format is supported");
        return *usable;
    }

    VkSurfaceCapabilitiesKHR surfaceCapabilities() const
    {
        VkSurfaceCapabilitiesKHR capabilities { };
        check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface.get(), &capabilities), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
        return capabilities;
    }

    static bool isDefined(VkExtent2D size)
    {
        return size.width != std::numeric_limits<uint32_t>::max() && size.width > 0 && size.height > 0;
    }

    VkExtent2D queryDisplayExtent() const
    {
        s32 width = 0, height = 0;
        Result rc = appletGetDefaultDisplayResolution(&width, &height);
        if (R_SUCCEEDED(rc) && width > 0 && height > 0)
            return { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

        Logger::warning("switch: failed to detect display resolution ({:#x}), falling back to the surface size", rc);
        VkSurfaceCapabilitiesKHR capabilities = surfaceCapabilities();
        if (isDefined(capabilities.currentExtent))
            return capabilities.currentExtent;
        if (isDefined(extent))
            return extent;
        return FallbackExtent;
    }

    VkPresentModeKHR choosePresentMode(int interval) const
    {
        if (interval >= 1)
            return VK_PRESENT_MODE_FIFO_KHR;

        auto modes = enumerate<VkPresentModeKHR>([&](uint32_t* count, VkPresentModeKHR* items)
            { vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface.get(), count, items); });
        for (VkPresentModeKHR preferred : { VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR })
            if (std::ranges::find(modes, preferred) != modes.end())
                return preferred;
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkSurfaceFormatKHR chooseSurfaceFormat() const
    {
        auto formats = enumerate<VkSurfaceFormatKHR>([&](uint32_t* count, VkSurfaceFormatKHR* items)
            { vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface.get(), count, items); });
        if (formats.empty())
            fatal("switch: the surface reports no formats");
        if (formats.size() == 1 && formats.front().format == VK_FORMAT_UNDEFINED)
            return { VK_FORMAT_R8G8B8A8_UNORM, formats.front().colorSpace };

        for (VkFormat preferred : { VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM })
        {
            auto match = std::ranges::find(formats, preferred, &VkSurfaceFormatKHR::format);
            if (match != formats.end())
                return *match;
        }
        return formats.front();
    }

    std::optional<uint32_t> findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required) const
    {
        auto indices = std::views::iota(0u, memoryProperties.memoryTypeCount);
        auto match   = std::ranges::find_if(indices, [&](uint32_t index)
            { return (typeBits & (1u << index)) && (memoryProperties.memoryTypes[index].propertyFlags & required) == required; });
        if (match == indices.end())
            return std::nullopt;
        return *match;
    }

    ImageView createView(VkImage image, VkFormat format, VkImageAspectFlags aspect) const
    {
        const VkImageViewCreateInfo info {
            .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image            = image,
            .viewType         = VK_IMAGE_VIEW_TYPE_2D,
            .format           = format,
            .subresourceRange = { aspect, 0, 1, 0, 1 },
        };
        VkImageView view = VK_NULL_HANDLE;
        check(vkCreateImageView(device.get(), &info, nullptr, &view), "vkCreateImageView");
        return ImageView(device.get(), view);
    }

    Semaphore createSemaphore() const
    {
        const VkSemaphoreCreateInfo info { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VkSemaphore semaphore = VK_NULL_HANDLE;
        check(vkCreateSemaphore(device.get(), &info, nullptr, &semaphore), "vkCreateSemaphore");
        return Semaphore(device.get(), semaphore);
    }

    Fence createSignaledFence() const
    {
        const VkFenceCreateInfo info { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT };
        VkFence fence = VK_NULL_HANDLE;
        check(vkCreateFence(device.get(), &info, nullptr, &fence), "vkCreateFence");
        return Fence(device.get(), fence);
    }

    void createSwapchain()
    {
        VkSurfaceCapabilitiesKHR capabilities = surfaceCapabilities();
        surfaceFormat                         = chooseSurfaceFormat();

        if (isDefined(capabilities.currentExtent))
            extent = capabilities.currentExtent;
        else
            extent = {
                std::clamp(desiredExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                std::clamp(desiredExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
            };

        uint32_t imageCount = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0)
            imageCount = std::min(imageCount, capabilities.maxImageCount);

        VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        if (!(capabilities.supportedCompositeAlpha & compositeAlpha))
            compositeAlpha = static_cast<VkCompositeAlphaFlagBitsKHR>(capabilities.supportedCompositeAlpha & -capabilities.supportedCompositeAlpha);

        VkSurfaceTransformFlagBitsKHR transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
        if (!(capabilities.supportedTransforms & transform))
            transform = capabilities.currentTransform;

        const VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
            | (capabilities.supportedUsageFlags & (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));

        const VkSwapchainCreateInfoKHR info {
            .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface          = surface.get(),
            .minImageCount    = imageCount,
            .imageFormat      = surfaceFormat.format,
            .imageColorSpace  = surfaceFormat.colorSpace,
            .imageExtent      = extent,
            .imageArrayLayers = 1,
            .imageUsage       = usage,
            .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .preTransform     = transform,
            .compositeAlpha   = compositeAlpha,
            .presentMode      = presentMode,
            .clipped          = VK_TRUE,
            .oldSwapchain     = swapchain.get(),
        };
        VkSwapchainKHR created = VK_NULL_HANDLE;
        check(vkCreateSwapchainKHR(device.get(), &info, nullptr, &created), "vkCreateSwapchainKHR");

        images.clear();
        swapchain = Swapchain(device.get(), created);

        auto handles = enumerate<VkImage>([&](uint32_t* count, VkImage* items)
            { vkGetSwapchainImagesKHR(device.get(), swapchain.get(), count, items); });
        images.reserve(handles.size());
        for (VkImage image : handles)
            images.push_back({ image, createView(image, surfaceFormat.format, VK_IMAGE_ASPECT_COLOR_BIT), createSemaphore() });

        if (!stencilImage.get() || stencilExtent.width != extent.width || stencilExtent.height != extent.height)
            createStencil();

        Logger::info("switch: vulkan swapchain {}x{}, {} images, format {}, present mode {}",
            extent.width, extent.height, images.size(), static_cast<int>(surfaceFormat.format), static_cast<int>(presentMode));
    }

    void createStencil()
    {
        stencilView.reset();
        stencilImage.reset();
        stencilMemory.reset();

        const VkImageCreateInfo info {
            .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType     = VK_IMAGE_TYPE_2D,
            .format        = stencilFormat,
            .extent        = { extent.width, extent.height, 1 },
            .mipLevels     = 1,
            .arrayLayers   = 1,
            .samples       = VK_SAMPLE_COUNT_1_BIT,
            .tiling        = VK_IMAGE_TILING_OPTIMAL,
            .usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        VkImage image = VK_NULL_HANDLE;
        check(vkCreateImage(device.get(), &info, nullptr, &image), "vkCreateImage");
        stencilImage = Image(device.get(), image);

        VkMemoryRequirements requirements;
        vkGetImageMemoryRequirements(device.get(), image, &requirements);
        auto type = findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!type)
            type = findMemoryType(requirements.memoryTypeBits, 0);
        if (!type)
            fatal("switch: no memory type for the stencil buffer");

        const VkMemoryAllocateInfo allocation {
            .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize  = requirements.size,
            .memoryTypeIndex = *type,
        };
        VkDeviceMemory memory = VK_NULL_HANDLE;
        check(vkAllocateMemory(device.get(), &allocation, nullptr, &memory), "vkAllocateMemory");
        stencilMemory = Memory(device.get(), memory);
        check(vkBindImageMemory(device.get(), image, memory, 0), "vkBindImageMemory");

        stencilView   = createView(image, stencilFormat, stencilAspect);
        stencilExtent = extent;
    }

    void createFrames()
    {
        for (Frame& frame : frames)
        {
            frame.fence         = createSignaledFence();
            frame.imageAcquired = createSemaphore();

            const VkCommandPoolCreateInfo poolInfo {
                .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                .queueFamilyIndex = queueFamilyIndex,
            };
            VkCommandPool pool = VK_NULL_HANDLE;
            check(vkCreateCommandPool(device.get(), &poolInfo, nullptr, &pool), "vkCreateCommandPool");
            frame.commandPool = CommandPool(device.get(), pool);

            const VkCommandBufferAllocateInfo allocation {
                .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool        = pool,
                .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 2,
            };
            std::array<VkCommandBuffer, 2> buffers { };
            check(vkAllocateCommandBuffers(device.get(), &allocation, buffers.data()), "vkAllocateCommandBuffers");
            frame.upload = buffers[0];
            frame.draw   = buffers[1];
        }
    }

    void waitIdle()
    {
        std::lock_guard lock(queueMutex);
        vkDeviceWaitIdle(device.get());
    }

    void runReleases(Frame& frame)
    {
        std::vector<std::function<void()>> pending;
        {
            std::lock_guard lock(releaseMutex);
            pending.swap(frame.releases);
        }
        for (auto& release : pending)
            release();
    }

    void recreateSwapchain()
    {
        waitIdle();
        presentMode         = requestedMode;
        VkExtent2D previous = extent;
        createSwapchain();
        recreateRequired = false;
        if (previous.width != extent.width || previous.height != extent.height)
            Application::onWindowResized(static_cast<int>(extent.width), static_cast<int>(extent.height));
    }

    void resetFramebuffer()
    {
        Logger::info("switch: resetting the vulkan framebuffer");
        waitIdle();
        desiredExtent  = queryDisplayExtent();
        resetRequested = false;
        recreateSwapchain();
    }

    std::optional<uint32_t> acquireImage(Frame& frame)
    {
        for (int attempt = 0; attempt < 2; attempt++)
        {
            uint32_t index  = 0;
            VkResult result = vkAcquireNextImageKHR(device.get(), swapchain.get(), std::numeric_limits<uint64_t>::max(), frame.imageAcquired.get(), VK_NULL_HANDLE, &index);
            if (result == VK_SUCCESS)
                return index;
            if (result == VK_SUBOPTIMAL_KHR)
            {
                recreateRequired = true;
                return index;
            }
            if (result != VK_ERROR_OUT_OF_DATE_KHR)
            {
                Logger::error("switch: vkAcquireNextImageKHR failed ({})", static_cast<int>(result));
                return std::nullopt;
            }
            recreateSwapchain();
        }
        return std::nullopt;
    }

    void transitionAttachmentsForFrame(VkCommandBuffer cmd) const
    {
        const std::array barriers {
            VkImageMemoryBarrier2 {
                .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .srcAccessMask    = VK_ACCESS_2_NONE,
                .dstStageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dstAccessMask    = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .image            = images[*imageIndex].image,
                .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
            },
            VkImageMemoryBarrier2 {
                .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask     = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                .srcAccessMask    = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                .dstStageMask     = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                .dstAccessMask    = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                .oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout        = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                .image            = stencilImage.get(),
                .subresourceRange = { stencilAspect, 0, 1, 0, 1 },
            },
        };
        const VkDependencyInfo dependency {
            .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
            .pImageMemoryBarriers    = barriers.data(),
        };
        vkCmdPipelineBarrier2(cmd, &dependency);
    }

    void orderAttachmentsForResume(VkCommandBuffer cmd) const
    {
        const VkMemoryBarrier2 barrier {
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        };
        const VkDependencyInfo dependency {
            .sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers    = &barrier,
        };
        vkCmdPipelineBarrier2(cmd, &dependency);
    }

    void transitionForPresent(VkCommandBuffer cmd) const
    {
        const VkImageMemoryBarrier2 barrier {
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask    = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dstStageMask     = VK_PIPELINE_STAGE_2_NONE,
            .dstAccessMask    = VK_ACCESS_2_NONE,
            .oldLayout        = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout        = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .image            = images[*imageIndex].image,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        const VkDependencyInfo dependency {
            .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers    = &barrier,
        };
        vkCmdPipelineBarrier2(cmd, &dependency);
    }

    void beginRendering(VkAttachmentLoadOp colorLoad)
    {
        if (!imageIndex || rendering)
            return;

        VkCommandBuffer cmd = frames[slot].draw;
        if (colorLoad == VK_ATTACHMENT_LOAD_OP_LOAD)
            orderAttachmentsForResume(cmd);

        const VkRenderingAttachmentInfo color {
            .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView   = images[*imageIndex].view.get(),
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp      = colorLoad,
            .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue  = { .color = { { 0.0f, 0.0f, 0.0f, 1.0f } } },
        };
        const VkRenderingAttachmentInfo stencil {
            .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView   = stencilView.get(),
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .loadOp      = colorLoad == VK_ATTACHMENT_LOAD_OP_LOAD ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue  = { .depthStencil = { 1.0f, 0 } },
        };
        const VkRenderingInfo info {
            .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea           = { { 0, 0 }, extent },
            .layerCount           = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &color,
            .pStencilAttachment   = &stencil,
        };
        vkCmdBeginRendering(cmd, &info);
        rendering = true;
    }

    void endRendering()
    {
        if (!rendering)
            return;
        vkCmdEndRendering(frames[slot].draw);
        rendering = false;
    }

    void beginFrame()
    {
        frameMutex.lock();
        {
            std::lock_guard lock(releaseMutex);
            slot = (slot + 1) % FramesInFlight;
        }

        if (resolutionEvent.poll() || resetRequested)
            resetFramebuffer();
        else if (recreateRequired || presentMode != requestedMode)
            recreateSwapchain();

        Frame& frame  = frames[slot];
        VkFence fence = frame.fence.get();
        check(vkWaitForFences(device.get(), 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max()), "vkWaitForFences");
        runReleases(frame);

        imageIndex = acquireImage(frame);
        check(vkResetFences(device.get(), 1, &fence), "vkResetFences");
        check(vkResetCommandPool(device.get(), frame.commandPool.get(), 0), "vkResetCommandPool");

        const VkCommandBufferBeginInfo begin {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        check(vkBeginCommandBuffer(frame.upload, &begin), "vkBeginCommandBuffer");
        check(vkBeginCommandBuffer(frame.draw, &begin), "vkBeginCommandBuffer");

        if (imageIndex)
        {
            transitionAttachmentsForFrame(frame.draw);
            beginRendering(VK_ATTACHMENT_LOAD_OP_CLEAR);
        }

        renderer->setFrame(frame.upload, imageIndex ? frame.draw : VK_NULL_HANDLE, slot);
    }

    void endFrame()
    {
        Frame& frame = frames[slot];
        endRendering();
        if (imageIndex)
            transitionForPresent(frame.draw);
        renderer->setFrame(VK_NULL_HANDLE, VK_NULL_HANDLE, slot);

        check(vkEndCommandBuffer(frame.upload), "vkEndCommandBuffer");
        check(vkEndCommandBuffer(frame.draw), "vkEndCommandBuffer");

        submit(frame);
        if (imageIndex)
            present();
        imageIndex.reset();

        frameMutex.unlock();
    }

    void submit(Frame& frame)
    {
        const VkSemaphoreSubmitInfo wait {
            .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = frame.imageAcquired.get(),
            .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        };
        const VkSemaphoreSubmitInfo signal {
            .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = imageIndex ? images[*imageIndex].renderFinished.get() : VK_NULL_HANDLE,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        };
        const std::array commandBuffers {
            VkCommandBufferSubmitInfo { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = frame.upload },
            VkCommandBufferSubmitInfo { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = frame.draw },
        };
        const uint32_t semaphoreCount = imageIndex ? 1 : 0;
        const VkSubmitInfo2 info {
            .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .waitSemaphoreInfoCount   = semaphoreCount,
            .pWaitSemaphoreInfos      = &wait,
            .commandBufferInfoCount   = static_cast<uint32_t>(commandBuffers.size()),
            .pCommandBufferInfos      = commandBuffers.data(),
            .signalSemaphoreInfoCount = semaphoreCount,
            .pSignalSemaphoreInfos    = &signal,
        };

        VkResult result;
        {
            std::lock_guard lock(queueMutex);
            result = vkQueueSubmit2(queue, 1, &info, frame.fence.get());
        }
        if (result != VK_SUCCESS)
        {
            Logger::error("switch: vkQueueSubmit2 failed ({})", static_cast<int>(result));
            waitIdle();
            frame.fence = createSignaledFence();
        }
    }

    void present()
    {
        const VkSemaphore wait     = images[*imageIndex].renderFinished.get();
        const VkSwapchainKHR chain = swapchain.get();
        const uint32_t index       = *imageIndex;
        const VkPresentInfoKHR info {
            .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = &wait,
            .swapchainCount     = 1,
            .pSwapchains        = &chain,
            .pImageIndices      = &index,
        };

        VkResult result;
        {
            std::lock_guard lock(queueMutex);
            result = vkQueuePresentKHR(queue, &info);
        }
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || recreateRequired)
            recreateSwapchain();
        else if (result != VK_SUCCESS)
            Logger::error("switch: vkQueuePresentKHR failed ({})", static_cast<int>(result));
    }

    void clear(NVGcolor color)
    {
        if (!rendering)
            return;

        const std::array attachments {
            VkClearAttachment {
                .aspectMask      = VK_IMAGE_ASPECT_COLOR_BIT,
                .colorAttachment = 0,
                .clearValue      = { .color = { { color.r, color.g, color.b, color.a } } },
            },
            VkClearAttachment {
                .aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT,
                .clearValue = { .depthStencil = { 1.0f, 0 } },
            },
        };
        const VkClearRect rect {
            .rect           = { { 0, 0 }, extent },
            .baseArrayLayer = 0,
            .layerCount     = 1,
        };
        vkCmdClearAttachments(frames[slot].draw, static_cast<uint32_t>(attachments.size()), attachments.data(), 1, &rect);
    }

    void defer(std::function<void()> release)
    {
        std::lock_guard lock(releaseMutex);
        frames[slot].releases.push_back(std::move(release));
    }
};

SwitchVulkanVideoContext::SwitchVulkanVideoContext()
    : state(std::make_unique<State>())
{
    Application::onWindowResized(static_cast<int>(state->extent.width), static_cast<int>(state->extent.height));
    Application::setWindowSize(static_cast<int>(state->extent.width), static_cast<int>(state->extent.height));
    setSwapInterval(VideoContext::swapInterval);
}

SwitchVulkanVideoContext::~SwitchVulkanVideoContext() = default;

void SwitchVulkanVideoContext::clear(NVGcolor color)
{
    state->clear(color);
}

void SwitchVulkanVideoContext::resetState()
{
}

void SwitchVulkanVideoContext::beginFrame()
{
    state->beginFrame();
}

void SwitchVulkanVideoContext::endFrame()
{
    state->endFrame();
}

void SwitchVulkanVideoContext::setSwapInterval(int interval)
{
    VideoContext::swapInterval = interval;
    state->requestedMode       = state->choosePresentMode(interval);
}

double SwitchVulkanVideoContext::getScaleFactor()
{
    return 1.0;
}

NVGcontext* SwitchVulkanVideoContext::getNVGContext()
{
    return state->nvgContext;
}

void SwitchVulkanVideoContext::appletCallback(AppletHookType hookType)
{
    if (hookType == AppletHookType_OnOperationMode && !state->resolutionEvent.isReady())
        state->resetRequested = true;
}

VkInstance SwitchVulkanVideoContext::instance() const
{
    return state->instance.get();
}

VkPhysicalDevice SwitchVulkanVideoContext::physicalDevice() const
{
    return state->physicalDevice;
}

VkDevice SwitchVulkanVideoContext::device() const
{
    return state->device.get();
}

VkQueue SwitchVulkanVideoContext::queue() const
{
    return state->queue;
}

uint32_t SwitchVulkanVideoContext::queueFamilyIndex() const
{
    return state->queueFamilyIndex;
}

std::mutex& SwitchVulkanVideoContext::queueMutex()
{
    return state->queueMutex;
}

const VkPhysicalDeviceMemoryProperties& SwitchVulkanVideoContext::memoryProperties() const
{
    return state->memoryProperties;
}

std::optional<uint32_t> SwitchVulkanVideoContext::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required) const
{
    return state->findMemoryType(typeBits, required);
}

bool SwitchVulkanVideoContext::hasDeviceExtension(std::string_view name) const
{
    return std::ranges::find(state->enabledExtensions, name) != state->enabledExtensions.end();
}

VkCommandBuffer SwitchVulkanVideoContext::uploadCommandBuffer() const
{
    return state->frames[state->slot].upload;
}

VkCommandBuffer SwitchVulkanVideoContext::frameCommandBuffer() const
{
    return state->frames[state->slot].draw;
}

uint32_t SwitchVulkanVideoContext::frameSlot() const
{
    return state->slot;
}

VkImage SwitchVulkanVideoContext::frameImage() const
{
    return state->imageIndex ? state->images[*state->imageIndex].image : VK_NULL_HANDLE;
}

VkImageView SwitchVulkanVideoContext::frameImageView() const
{
    return state->imageIndex ? state->images[*state->imageIndex].view.get() : VK_NULL_HANDLE;
}

VkExtent2D SwitchVulkanVideoContext::frameExtent() const
{
    return state->extent;
}

VkFormat SwitchVulkanVideoContext::colorFormat() const
{
    return state->surfaceFormat.format;
}

VkFormat SwitchVulkanVideoContext::stencilFormat() const
{
    return state->stencilFormat;
}

void SwitchVulkanVideoContext::beginRendering(VkAttachmentLoadOp colorLoad)
{
    state->beginRendering(colorLoad);
}

void SwitchVulkanVideoContext::endRendering()
{
    state->endRendering();
}

bool SwitchVulkanVideoContext::isRendering() const
{
    return state->rendering;
}

void SwitchVulkanVideoContext::deferUntilFrameRetired(std::function<void()> release)
{
    state->defer(std::move(release));
}

} // namespace brls

#endif
