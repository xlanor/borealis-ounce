#pragma once

#include <switch.h>
#include <vulkan/vulkan.h>

#include <borealis/core/video.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>

namespace brls
{

class SwitchVulkanVideoContext : public VideoContext
{
  public:
    static constexpr uint32_t FramesInFlight = 2;

    SwitchVulkanVideoContext();
    ~SwitchVulkanVideoContext() override;
    SwitchVulkanVideoContext(const SwitchVulkanVideoContext&)            = delete;
    SwitchVulkanVideoContext& operator=(const SwitchVulkanVideoContext&) = delete;

    void clear(NVGcolor color) override;
    void resetState() override;
    void beginFrame() override;
    void endFrame() override;
    void setSwapInterval(int interval) override;
    double getScaleFactor() override;
    NVGcontext* getNVGContext() override;

    void appletCallback(AppletHookType hookType);

    VkInstance instance() const;
    VkPhysicalDevice physicalDevice() const;
    VkDevice device() const;
    VkQueue queue() const;
    uint32_t queueFamilyIndex() const;
    std::mutex& queueMutex();
    const VkPhysicalDeviceMemoryProperties& memoryProperties() const;
    std::optional<uint32_t> findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required) const;
    bool hasDeviceExtension(std::string_view name) const;

    VkCommandBuffer uploadCommandBuffer() const;
    VkCommandBuffer frameCommandBuffer() const;
    uint32_t frameSlot() const;
    VkImage frameImage() const;
    VkImageView frameImageView() const;
    VkExtent2D frameExtent() const;
    VkFormat colorFormat() const;
    VkFormat stencilFormat() const;

    void beginRendering(VkAttachmentLoadOp colorLoad);
    void endRendering();
    bool isRendering() const;

    void deferUntilFrameRetired(std::function<void()> release);

  private:
    struct State;
    std::unique_ptr<State> state;
};

#ifndef BOREALIS_USE_DEKO3D
using SwitchVideoContext = SwitchVulkanVideoContext;
#endif

} // namespace brls
