#include <algorithm>
#include <array>
#include <bit>
#include <borealis/core/logger.hpp>
#include <borealis/core/vk_handle.hpp>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <map>
#include <nanovg/vk_renderer.hpp>
#include <optional>
#include <ranges>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nanovg_vk_fill_frag.h"
#include "nanovg_vk_fill_vert.h"

namespace nvg
{

struct VkRenderer::State
{
    using Buffer              = brls::vk::DeviceHandle<VkBuffer, vkDestroyBuffer>;
    using Memory              = brls::vk::DeviceHandle<VkDeviceMemory, vkFreeMemory>;
    using Image               = brls::vk::DeviceHandle<VkImage, vkDestroyImage>;
    using ImageView           = brls::vk::DeviceHandle<VkImageView, vkDestroyImageView>;
    using Sampler             = brls::vk::DeviceHandle<VkSampler, vkDestroySampler>;
    using DescriptorSetLayout = brls::vk::DeviceHandle<VkDescriptorSetLayout, vkDestroyDescriptorSetLayout>;
    using DescriptorPool      = brls::vk::DeviceHandle<VkDescriptorPool, vkDestroyDescriptorPool>;
    using PipelineLayout      = brls::vk::DeviceHandle<VkPipelineLayout, vkDestroyPipelineLayout>;
    using Pipeline            = brls::vk::DeviceHandle<VkPipeline, vkDestroyPipeline>;
    using ShaderModule        = brls::vk::DeviceHandle<VkShaderModule, vkDestroyShaderModule>;

    class DescriptorSet
    {
      public:
        DescriptorSet(VkDevice device, VkDescriptorPool pool, VkDescriptorSet set)
            : device(device)
            , pool(pool)
            , set(set)
        {
        }
        ~DescriptorSet()
        {
            if (set != VK_NULL_HANDLE)
                vkFreeDescriptorSets(device, pool, 1, &set);
        }
        DescriptorSet(DescriptorSet&& other) noexcept
            : device(other.device)
            , pool(other.pool)
            , set(std::exchange(other.set, VK_NULL_HANDLE))
        {
        }
        DescriptorSet& operator=(DescriptorSet&& other) noexcept
        {
            if (this != &other)
            {
                if (set != VK_NULL_HANDLE)
                    vkFreeDescriptorSets(device, pool, 1, &set);
                device = other.device;
                pool   = other.pool;
                set    = std::exchange(other.set, VK_NULL_HANDLE);
            }
            return *this;
        }

        VkDescriptorSet get() const { return set; }

      private:
        VkDevice device;
        VkDescriptorPool pool;
        VkDescriptorSet set;
    };

    class HostBuffer
    {
      public:
        HostBuffer(Buffer buffer, Memory memory, std::byte* mapped, VkDeviceSize size)
            : memory(std::move(memory))
            , buffer(std::move(buffer))
            , mapped(mapped)
            , capacity(size)
        {
        }

        VkBuffer get() const { return buffer.get(); }
        std::byte* data() const { return mapped; }
        VkDeviceSize size() const { return capacity; }

      private:
        Memory memory;
        Buffer buffer;
        std::byte* mapped;
        VkDeviceSize capacity;
    };

    class Texture;

    struct Graveyard
    {
        std::vector<HostBuffer> buffers;
        std::vector<DescriptorSet> sets;
        std::vector<std::unique_ptr<Texture>> textures;

        void clear()
        {
            textures.clear();
            sets.clear();
            buffers.clear();
        }
    };

    struct Allocation
    {
        VkBuffer buffer;
        VkDeviceSize offset;
        std::byte* data;
    };

    class Arena
    {
      public:
        explicit Arena(VkBufferUsageFlags usage)
            : usage(usage)
        {
        }

        void reset() { used = 0; }

        VkBuffer buffer() const { return current ? current->get() : VK_NULL_HANDLE; }

        std::optional<Allocation> allocate(const State& state, Graveyard& graveyard, VkDeviceSize size, VkDeviceSize alignment)
        {
            VkDeviceSize offset = alignUp(used, alignment);
            if (!current || offset + size > current->size())
            {
                VkDeviceSize capacity = std::max<VkDeviceSize>({ size, current ? current->size() * 2 : 0, MinimumCapacity });
                auto replacement      = state.createHostBuffer(capacity, usage);
                if (!replacement)
                    return std::nullopt;
                if (current)
                    graveyard.buffers.push_back(std::move(*current));
                current = std::move(replacement);
                offset  = 0;
            }
            used = offset + size;
            return Allocation { current->get(), offset, current->data() + offset };
        }

      private:
        static constexpr VkDeviceSize MinimumCapacity = 64 * 1024;

        VkBufferUsageFlags usage;
        std::optional<HostBuffer> current;
        VkDeviceSize used = 0;
    };

    struct Frame
    {
        Arena vertices { VK_BUFFER_USAGE_VERTEX_BUFFER_BIT };
        Arena uniforms { VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT };
        Arena staging { VK_BUFFER_USAGE_TRANSFER_SRC_BIT };
        Graveyard graveyard;
        std::optional<DescriptorSet> uniformSet;
        VkBuffer uniformSetBuffer = VK_NULL_HANDLE;
    };

    class Texture
    {
      public:
        struct Description
        {
            int width;
            int height;
            int type;
            int flags;
        };

        Texture(Description description, Image image, Memory memory, ImageView view, DescriptorSet descriptor, uint32_t mipLevels)
            : desc(description)
            , memory(std::move(memory))
            , image(std::move(image))
            , ownedView(std::move(view))
            , descriptor(std::move(descriptor))
            , mipLevels(mipLevels)
            , layout(VK_IMAGE_LAYOUT_UNDEFINED)
            , needsClear(true)
        {
        }

        Texture(Description description, DescriptorSet descriptor)
            : desc(description)
            , descriptor(std::move(descriptor))
            , mipLevels(1)
            , layout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            , needsClear(false)
        {
        }

        const Description& description() const { return desc; }
        VkDescriptorSet descriptorSet() const { return descriptor.get(); }
        bool isExternal() const { return image.get() == VK_NULL_HANDLE; }

        void queueUpload(int x, int y, int w, int h, const unsigned char* data)
        {
            const int left   = std::clamp(x, 0, desc.width);
            const int top    = std::clamp(y, 0, desc.height);
            const int right  = std::clamp(x + w, left, desc.width);
            const int bottom = std::clamp(y + h, top, desc.height);
            if (isExternal() || data == nullptr || right == left || bottom == top)
                return;

            const size_t pixel = bytesPerPixel();
            const size_t row   = static_cast<size_t>(right - left) * pixel;
            Upload upload {
                .offset = { left, top },
                .extent = { static_cast<uint32_t>(right - left), static_cast<uint32_t>(bottom - top) },
                .pixels = std::vector<unsigned char>(row * static_cast<size_t>(bottom - top)),
            };
            for (int line : std::views::iota(top, bottom))
            {
                const unsigned char* source = data + (static_cast<size_t>(line) * static_cast<size_t>(desc.width) + static_cast<size_t>(left)) * pixel;
                std::memcpy(upload.pixels.data() + static_cast<size_t>(line - top) * row, source, row);
            }

            if (upload.offset.x == 0 && upload.offset.y == 0 && upload.extent.width == static_cast<uint32_t>(desc.width) && upload.extent.height == static_cast<uint32_t>(desc.height))
            {
                uploads.clear();
                needsClear = false;
            }
            uploads.push_back(std::move(upload));
        }

        void recordUploads(VkCommandBuffer cmd, State& state)
        {
            if (!needsClear && uploads.empty())
                return;

            const VkImageSubresourceRange all { VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1 };
            const bool sampled = layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageBarrier(cmd, all, layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                sampled ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

            bool written = false;
            if (needsClear)
            {
                const VkClearColorValue zero { };
                vkCmdClearColorImage(cmd, image.get(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &all);
                written = true;
            }

            for (const Upload& upload : uploads)
            {
                auto staged = state.stage(upload.pixels.size());
                if (!staged)
                {
                    brls::Logger::error("nanovg: failed to stage a {}x{} texture upload", upload.extent.width, upload.extent.height);
                    continue;
                }
                std::memcpy(staged->data, upload.pixels.data(), upload.pixels.size());

                if (written)
                    transferBarrier(cmd);

                const VkBufferImageCopy region {
                    .bufferOffset     = staged->offset,
                    .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                    .imageOffset      = { upload.offset.x, upload.offset.y, 0 },
                    .imageExtent      = { upload.extent.width, upload.extent.height, 1 },
                };
                vkCmdCopyBufferToImage(cmd, staged->buffer, image.get(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
                written = true;
            }

            if (mipLevels > 1)
                generateMipmaps(cmd);
            else
                imageBarrier(cmd, all, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

            layout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            needsClear = false;
            uploads.clear();
        }

      private:
        struct Upload
        {
            VkOffset2D offset;
            VkExtent2D extent;
            std::vector<unsigned char> pixels;
        };

        size_t bytesPerPixel() const { return desc.type == NVG_TEXTURE_RGBA ? 4 : 1; }

        void imageBarrier(VkCommandBuffer cmd, VkImageSubresourceRange range, VkImageLayout from, VkImageLayout to,
            VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) const
        {
            const VkImageMemoryBarrier2 barrier {
                .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .srcStageMask        = srcStage,
                .srcAccessMask       = srcAccess,
                .dstStageMask        = dstStage,
                .dstAccessMask       = dstAccess,
                .oldLayout           = from,
                .newLayout           = to,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image               = image.get(),
                .subresourceRange    = range,
            };
            const VkDependencyInfo dependency {
                .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .imageMemoryBarrierCount = 1,
                .pImageMemoryBarriers    = &barrier,
            };
            vkCmdPipelineBarrier2(cmd, &dependency);
        }

        static void transferBarrier(VkCommandBuffer cmd)
        {
            const VkMemoryBarrier2 barrier {
                .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .srcStageMask  = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .dstStageMask  = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            };
            const VkDependencyInfo dependency {
                .sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .memoryBarrierCount = 1,
                .pMemoryBarriers    = &barrier,
            };
            vkCmdPipelineBarrier2(cmd, &dependency);
        }

        void generateMipmaps(VkCommandBuffer cmd) const
        {
            transferBarrier(cmd);
            for (uint32_t level : std::views::iota(1u, mipLevels))
            {
                imageBarrier(cmd, { VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 1, 0, 1 },
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

                const VkImageBlit blit {
                    .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1 },
                    .srcOffsets     = { { 0, 0, 0 }, { mipSize(desc.width, level - 1), mipSize(desc.height, level - 1), 1 } },
                    .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1 },
                    .dstOffsets     = { { 0, 0, 0 }, { mipSize(desc.width, level), mipSize(desc.height, level), 1 } },
                };
                vkCmdBlitImage(cmd, image.get(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image.get(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
            }

            imageBarrier(cmd, { VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels - 1, 0, 1 },
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            imageBarrier(cmd, { VK_IMAGE_ASPECT_COLOR_BIT, mipLevels - 1, 1, 0, 1 },
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }

        static int32_t mipSize(int size, uint32_t level)
        {
            return std::max(1, size >> level);
        }

        Description desc;
        Memory memory;
        Image image;
        ImageView ownedView;
        DescriptorSet descriptor;
        uint32_t mipLevels;
        VkImageLayout layout;
        bool needsClear;
        std::vector<Upload> uploads;
    };

    enum class CallType
    {
        Fill,
        ConvexFill,
        Stroke,
        Triangles,
    };

    enum ShaderType : int32_t
    {
        ShaderFillGradient = 0,
        ShaderFillImage    = 1,
        ShaderSimple       = 2,
        ShaderImage        = 3,
    };

    struct FragUniforms
    {
        std::array<float, 12> scissorMat;
        std::array<float, 12> paintMat;
        NVGcolor innerCol;
        NVGcolor outerCol;
        std::array<float, 2> scissorExt;
        std::array<float, 2> scissorScale;
        std::array<float, 2> extent;
        float radius;
        float feather;
        float strokeMult;
        float strokeThr;
        int32_t texType;
        int32_t type;
    };
    static_assert(sizeof(FragUniforms) == 176);

    struct Blend
    {
        VkBlendFactor srcRGB;
        VkBlendFactor dstRGB;
        VkBlendFactor srcAlpha;
        VkBlendFactor dstAlpha;

        auto operator<=>(const Blend&) const = default;
    };

    struct PipelineKey
    {
        Blend blend;
        bool colorWrite;
        bool antialias;

        auto operator<=>(const PipelineKey&) const = default;
    };

    struct Path
    {
        size_t fillOffset   = 0;
        size_t fillCount    = 0;
        size_t strokeOffset = 0;
        size_t strokeCount  = 0;
    };

    struct Call
    {
        CallType type;
        int image;
        size_t pathOffset;
        size_t pathCount;
        size_t triangleOffset;
        size_t triangleCount;
        size_t uniformOffset;
        Blend blend;
    };

    enum class StencilMode
    {
        Off,
        FillWinding,
        Match,
        CoverClear,
        StrokeBase,
        StrokeClear,
    };

    class Context
    {
      public:
        Context(State& state, int flags)
            : state(state)
            , flags(flags)
        {
        }

        static NVGparams params(Context* context)
        {
            NVGparams params { };
            params.userPtr              = context;
            params.edgeAntiAlias        = (context->flags & NVG_ANTIALIAS) ? 1 : 0;
            params.renderCreate         = &Context::renderCreate;
            params.renderCreateTexture  = &Context::renderCreateTexture;
            params.renderDeleteTexture  = &Context::renderDeleteTexture;
            params.renderUpdateTexture  = &Context::renderUpdateTexture;
            params.renderGetTextureSize = &Context::renderGetTextureSize;
            params.renderViewport       = &Context::renderViewport;
            params.renderCancel         = &Context::renderCancel;
            params.renderFlush          = &Context::renderFlush;
            params.renderFill           = &Context::renderFill;
            params.renderStroke         = &Context::renderStroke;
            params.renderTriangles      = &Context::renderTriangles;
            params.renderDelete         = &Context::renderDelete;
            return params;
        }

        bool antialias() const { return flags & NVG_ANTIALIAS; }
        bool stencilStrokes() const { return flags & NVG_STENCIL_STROKES; }
        std::array<float, 2> viewSize() const { return view; }
        float devicePixelRatio() const { return pixelRatio; }
        std::span<const Call> recordedCalls() const { return calls; }
        std::span<const Path> recordedPaths() const { return paths; }
        std::span<const NVGvertex> recordedVertices() const { return vertices; }
        std::span<const FragUniforms> recordedUniforms() const { return uniforms; }

        void reset()
        {
            calls.clear();
            paths.clear();
            vertices.clear();
            uniforms.clear();
        }

      private:
        static Context& from(void* userPtr) { return *static_cast<Context*>(userPtr); }

        static int renderCreate(void*) { return 1; }

        static int renderCreateTexture(void* userPtr, int type, int w, int h, int imageFlags, const unsigned char* data)
        {
            return from(userPtr).state.createTexture(type, w, h, imageFlags, data);
        }

        static int renderDeleteTexture(void* userPtr, int image)
        {
            return from(userPtr).state.deleteTexture(image) ? 1 : 0;
        }

        static int renderUpdateTexture(void* userPtr, int image, int x, int y, int w, int h, const unsigned char* data)
        {
            return from(userPtr).state.updateTexture(image, x, y, w, h, data) ? 1 : 0;
        }

        static int renderGetTextureSize(void* userPtr, int image, int* w, int* h)
        {
            const Texture* texture = from(userPtr).state.findTexture(image);
            if (!texture)
                return 0;
            *w = texture->description().width;
            *h = texture->description().height;
            return 1;
        }

        static void renderViewport(void* userPtr, float width, float height, float devicePixelRatio)
        {
            Context& context   = from(userPtr);
            context.view       = { width, height };
            context.pixelRatio = devicePixelRatio;
        }

        static void renderCancel(void* userPtr)
        {
            from(userPtr).reset();
        }

        static void renderFlush(void* userPtr)
        {
            Context& context = from(userPtr);
            context.state.flush(context);
            context.reset();
        }

        static void renderFill(void* userPtr, NVGpaint* paint, NVGcompositeOperationState compositeOperation, NVGscissor* scissor, float fringe, const float* bounds, const NVGpath* paths, int npaths)
        {
            Context& context = from(userPtr);
            std::span<const NVGpath> source(paths, static_cast<size_t>(npaths));
            const bool convex = source.size() == 1 && source.front().convex;

            Call call {
                .type       = convex ? CallType::ConvexFill : CallType::Fill,
                .image      = paint->image,
                .pathOffset = context.paths.size(),
                .pathCount  = source.size(),
                .blend      = blendFor(compositeOperation),
            };

            for (const NVGpath& path : source)
                context.paths.push_back(context.copyPath(path, true));

            if (call.type == CallType::Fill)
            {
                call.triangleOffset = context.vertices.size();
                call.triangleCount  = 4;
                context.vertices.push_back({ bounds[2], bounds[3], 0.5f, 1.0f });
                context.vertices.push_back({ bounds[2], bounds[1], 0.5f, 1.0f });
                context.vertices.push_back({ bounds[0], bounds[3], 0.5f, 1.0f });
                context.vertices.push_back({ bounds[0], bounds[1], 0.5f, 1.0f });

                call.uniformOffset = context.uniforms.size();
                FragUniforms simple { };
                simple.strokeThr = -1.0f;
                simple.type      = ShaderSimple;
                context.uniforms.push_back(simple);
                context.uniforms.push_back(context.convertPaint(*paint, *scissor, fringe, fringe, -1.0f));
            }
            else
            {
                call.uniformOffset = context.uniforms.size();
                context.uniforms.push_back(context.convertPaint(*paint, *scissor, fringe, fringe, -1.0f));
            }

            context.calls.push_back(call);
        }

        static void renderStroke(void* userPtr, NVGpaint* paint, NVGcompositeOperationState compositeOperation, NVGscissor* scissor, float fringe, float strokeWidth, const NVGpath* paths, int npaths)
        {
            Context& context = from(userPtr);
            std::span<const NVGpath> source(paths, static_cast<size_t>(npaths));

            Call call {
                .type          = CallType::Stroke,
                .image         = paint->image,
                .pathOffset    = context.paths.size(),
                .pathCount     = source.size(),
                .uniformOffset = context.uniforms.size(),
                .blend         = blendFor(compositeOperation),
            };

            for (const NVGpath& path : source)
                context.paths.push_back(context.copyPath(path, false));

            context.uniforms.push_back(context.convertPaint(*paint, *scissor, strokeWidth, fringe, -1.0f));
            if (context.stencilStrokes())
                context.uniforms.push_back(context.convertPaint(*paint, *scissor, strokeWidth, fringe, 1.0f - 0.5f / 255.0f));

            context.calls.push_back(call);
        }

        static void renderTriangles(void* userPtr, NVGpaint* paint, NVGcompositeOperationState compositeOperation, NVGscissor* scissor, const NVGvertex* verts, int nverts, float fringe)
        {
            Context& context = from(userPtr);
            std::span<const NVGvertex> source(verts, static_cast<size_t>(nverts));

            Call call {
                .type           = CallType::Triangles,
                .image          = paint->image,
                .triangleOffset = context.vertices.size(),
                .triangleCount  = source.size(),
                .uniformOffset  = context.uniforms.size(),
                .blend          = blendFor(compositeOperation),
            };
            context.vertices.insert(context.vertices.end(), source.begin(), source.end());

            FragUniforms frag = context.convertPaint(*paint, *scissor, 1.0f, fringe, -1.0f);
            frag.type         = ShaderImage;
            context.uniforms.push_back(frag);

            context.calls.push_back(call);
        }

        static void renderDelete(void* userPtr)
        {
            delete static_cast<Context*>(userPtr);
        }

        Path copyPath(const NVGpath& path, bool includeFill)
        {
            Path copy;
            if (includeFill && path.nfill > 0)
            {
                copy.fillOffset = vertices.size();
                copy.fillCount  = static_cast<size_t>(path.nfill);
                vertices.insert(vertices.end(), path.fill, path.fill + path.nfill);
            }
            if (path.nstroke > 0)
            {
                copy.strokeOffset = vertices.size();
                copy.strokeCount  = static_cast<size_t>(path.nstroke);
                vertices.insert(vertices.end(), path.stroke, path.stroke + path.nstroke);
            }
            return copy;
        }

        static std::array<float, 12> toMat3x4(const std::array<float, 6>& t)
        {
            return { t[0], t[1], 0.0f, 0.0f, t[2], t[3], 0.0f, 0.0f, t[4], t[5], 1.0f, 0.0f };
        }

        static NVGcolor premultiplied(NVGcolor color)
        {
            color.r *= color.a;
            color.g *= color.a;
            color.b *= color.a;
            return color;
        }

        FragUniforms convertPaint(const NVGpaint& paint, const NVGscissor& scissor, float width, float fringe, float strokeThr) const
        {
            FragUniforms frag { };
            frag.innerCol = premultiplied(paint.innerColor);
            frag.outerCol = premultiplied(paint.outerColor);

            std::array<float, 6> inverse { };
            if (scissor.extent[0] < -0.5f || scissor.extent[1] < -0.5f)
            {
                frag.scissorMat   = { };
                frag.scissorExt   = { 1.0f, 1.0f };
                frag.scissorScale = { 1.0f, 1.0f };
            }
            else
            {
                nvgTransformInverse(inverse.data(), scissor.xform);
                frag.scissorMat   = toMat3x4(inverse);
                frag.scissorExt   = { scissor.extent[0], scissor.extent[1] };
                frag.scissorScale = {
                    std::sqrt(scissor.xform[0] * scissor.xform[0] + scissor.xform[2] * scissor.xform[2]) / fringe,
                    std::sqrt(scissor.xform[1] * scissor.xform[1] + scissor.xform[3] * scissor.xform[3]) / fringe,
                };
            }

            frag.extent     = { paint.extent[0], paint.extent[1] };
            frag.strokeMult = (width * 0.5f + fringe * 0.5f) / fringe;
            frag.strokeThr  = strokeThr;

            if (paint.image != 0)
            {
                const Texture* texture = state.findTexture(paint.image);
                if (!texture)
                    return frag;

                const Texture::Description& description = texture->description();
                if (description.flags & NVG_IMAGE_FLIPY)
                {
                    std::array<float, 6> m1 { }, m2 { };
                    nvgTransformTranslate(m1.data(), 0.0f, frag.extent[1] * 0.5f);
                    nvgTransformMultiply(m1.data(), paint.xform);
                    nvgTransformScale(m2.data(), 1.0f, -1.0f);
                    nvgTransformMultiply(m2.data(), m1.data());
                    nvgTransformTranslate(m1.data(), 0.0f, -frag.extent[1] * 0.5f);
                    nvgTransformMultiply(m1.data(), m2.data());
                    nvgTransformInverse(inverse.data(), m1.data());
                }
                else
                {
                    nvgTransformInverse(inverse.data(), paint.xform);
                }

                frag.type = ShaderFillImage;
                if (description.type == NVG_TEXTURE_RGBA)
                    frag.texType = (description.flags & NVG_IMAGE_PREMULTIPLIED) ? 0 : 1;
                else
                    frag.texType = 2;
            }
            else
            {
                frag.type    = ShaderFillGradient;
                frag.radius  = paint.radius;
                frag.feather = paint.feather;
                nvgTransformInverse(inverse.data(), paint.xform);
            }

            frag.paintMat = toMat3x4(inverse);
            return frag;
        }

        static std::optional<VkBlendFactor> blendFactor(int factor)
        {
            switch (factor)
            {
                case NVG_ZERO:
                    return VK_BLEND_FACTOR_ZERO;
                case NVG_ONE:
                    return VK_BLEND_FACTOR_ONE;
                case NVG_SRC_COLOR:
                    return VK_BLEND_FACTOR_SRC_COLOR;
                case NVG_ONE_MINUS_SRC_COLOR:
                    return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
                case NVG_DST_COLOR:
                    return VK_BLEND_FACTOR_DST_COLOR;
                case NVG_ONE_MINUS_DST_COLOR:
                    return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
                case NVG_SRC_ALPHA:
                    return VK_BLEND_FACTOR_SRC_ALPHA;
                case NVG_ONE_MINUS_SRC_ALPHA:
                    return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                case NVG_DST_ALPHA:
                    return VK_BLEND_FACTOR_DST_ALPHA;
                case NVG_ONE_MINUS_DST_ALPHA:
                    return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
                case NVG_SRC_ALPHA_SATURATE:
                    return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
                default:
                    return std::nullopt;
            }
        }

        static Blend blendFor(NVGcompositeOperationState op)
        {
            auto srcRGB   = blendFactor(op.srcRGB);
            auto dstRGB   = blendFactor(op.dstRGB);
            auto srcAlpha = blendFactor(op.srcAlpha);
            auto dstAlpha = blendFactor(op.dstAlpha);
            if (!srcRGB || !dstRGB || !srcAlpha || !dstAlpha)
                return DefaultBlend;
            return { *srcRGB, *dstRGB, *srcAlpha, *dstAlpha };
        }

        State& state;
        int flags;
        std::array<float, 2> view { };
        float pixelRatio = 1.0f;
        std::vector<Call> calls;
        std::vector<Path> paths;
        std::vector<NVGvertex> vertices;
        std::vector<FragUniforms> uniforms;
    };

    class Recorder
    {
      public:
        Recorder(State& state, const Context& context, VkCommandBuffer cmd, VkDescriptorSet uniformSet, VkDeviceSize uniformBase)
            : state(state)
            , context(context)
            , cmd(cmd)
            , uniformSet(uniformSet)
            , uniformBase(uniformBase)
        {
        }

        void record(const Call& call)
        {
            switch (call.type)
            {
                case CallType::Fill:
                    fill(call);
                    break;
                case CallType::ConvexFill:
                    convexFill(call);
                    break;
                case CallType::Stroke:
                    stroke(call);
                    break;
                case CallType::Triangles:
                    triangles(call);
                    break;
            }
        }

      private:
        std::span<const Path> pathsOf(const Call& call) const
        {
            return context.recordedPaths().subspan(call.pathOffset, call.pathCount);
        }

        bool bindPipeline(const Blend& blend, bool colorWrite)
        {
            VkPipeline pipeline = state.pipeline({ colorWrite ? blend : DefaultBlend, colorWrite, context.antialias() });
            if (pipeline == VK_NULL_HANDLE)
                return false;
            if (pipeline != bound)
            {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                bound = pipeline;
            }
            return true;
        }

        void bindPaint(size_t uniformIndex, int image)
        {
            const uint32_t offset = static_cast<uint32_t>(uniformBase + uniformIndex * state.uniformStride);
            const std::array sets { uniformSet, state.textureSet(image) };
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, state.pipelineLayout.get(), 0,
                static_cast<uint32_t>(sets.size()), sets.data(), 1, &offset);
        }

        void stencil(StencilMode mode)
        {
            if (mode == StencilMode::Off)
            {
                vkCmdSetStencilTestEnable(cmd, VK_FALSE);
                return;
            }

            vkCmdSetStencilTestEnable(cmd, VK_TRUE);
            switch (mode)
            {
                case StencilMode::FillWinding:
                    vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_FRONT_BIT, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_INCREMENT_AND_WRAP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
                    vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_BACK_BIT, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_DECREMENT_AND_WRAP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
                    break;
                case StencilMode::Match:
                    vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_EQUAL);
                    break;
                case StencilMode::CoverClear:
                    vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_ZERO, VK_STENCIL_OP_ZERO, VK_STENCIL_OP_ZERO, VK_COMPARE_OP_NOT_EQUAL);
                    break;
                case StencilMode::StrokeBase:
                    vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_INCREMENT_AND_CLAMP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_EQUAL);
                    break;
                case StencilMode::StrokeClear:
                    vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_ZERO, VK_STENCIL_OP_ZERO, VK_STENCIL_OP_ZERO, VK_COMPARE_OP_ALWAYS);
                    break;
                case StencilMode::Off:
                    break;
            }
        }

        void draw(VkPrimitiveTopology topology, size_t count, size_t first)
        {
            if (count == 0)
                return;
            vkCmdSetPrimitiveTopology(cmd, topology);
            vkCmdDraw(cmd, static_cast<uint32_t>(count), 1, static_cast<uint32_t>(first), 0);
        }

        void drawFills(std::span<const Path> paths)
        {
            for (const Path& path : paths)
                draw(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN, path.fillCount, path.fillOffset);
        }

        void drawStrokes(std::span<const Path> paths)
        {
            for (const Path& path : paths)
                draw(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, path.strokeCount, path.strokeOffset);
        }

        void fill(const Call& call)
        {
            auto paths = pathsOf(call);

            if (!bindPipeline(call.blend, false))
                return;
            bindPaint(call.uniformOffset, 0);
            stencil(StencilMode::FillWinding);
            vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE);
            drawFills(paths);

            vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
            if (bindPipeline(call.blend, true))
            {
                bindPaint(call.uniformOffset + 1, call.image);
                if (context.antialias())
                {
                    stencil(StencilMode::Match);
                    drawStrokes(paths);
                }
                stencil(StencilMode::CoverClear);
                draw(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, call.triangleCount, call.triangleOffset);
            }
            stencil(StencilMode::Off);
        }

        void convexFill(const Call& call)
        {
            if (!bindPipeline(call.blend, true))
                return;
            bindPaint(call.uniformOffset, call.image);
            for (const Path& path : pathsOf(call))
            {
                draw(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN, path.fillCount, path.fillOffset);
                draw(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, path.strokeCount, path.strokeOffset);
            }
        }

        void stroke(const Call& call)
        {
            auto paths = pathsOf(call);

            if (!context.stencilStrokes())
            {
                if (!bindPipeline(call.blend, true))
                    return;
                bindPaint(call.uniformOffset, call.image);
                drawStrokes(paths);
                return;
            }

            if (bindPipeline(call.blend, true))
            {
                stencil(StencilMode::StrokeBase);
                bindPaint(call.uniformOffset + 1, call.image);
                drawStrokes(paths);

                stencil(StencilMode::Match);
                bindPaint(call.uniformOffset, call.image);
                drawStrokes(paths);
            }

            if (bindPipeline(call.blend, false))
            {
                stencil(StencilMode::StrokeClear);
                drawStrokes(paths);
            }
            stencil(StencilMode::Off);
        }

        void triangles(const Call& call)
        {
            if (!bindPipeline(call.blend, true))
                return;
            bindPaint(call.uniformOffset, call.image);
            draw(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, call.triangleCount, call.triangleOffset);
        }

        State& state;
        const Context& context;
        VkCommandBuffer cmd;
        VkDescriptorSet uniformSet;
        VkDeviceSize uniformBase;
        VkPipeline bound = VK_NULL_HANDLE;
    };

    static constexpr Blend DefaultBlend { VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA };
    static constexpr uint32_t SamplerCount = 16;
    static constexpr uint32_t SetsPerPool  = 256;

    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkPhysicalDeviceMemoryProperties memoryProperties;
    Target target;
    VkDeviceSize uniformStride    = 0;
    VkDeviceSize stagingAlignment = 16;

    DescriptorSetLayout uniformLayout;
    DescriptorSetLayout textureLayout;
    PipelineLayout pipelineLayout;
    ShaderModule vertexShader;
    ShaderModule fragmentShader;
    std::array<Sampler, SamplerCount> samplers;
    std::map<PipelineKey, Pipeline> pipelines;
    std::vector<DescriptorPool> descriptorPools;

    std::vector<Frame> frames;
    uint32_t slot           = 0;
    bool frameActive        = false;
    VkCommandBuffer upload  = VK_NULL_HANDLE;
    VkCommandBuffer drawCmd = VK_NULL_HANDLE;

    std::unordered_map<int, std::unique_ptr<Texture>> textures;
    int nextTextureId = 1;
    std::unique_ptr<Texture> fallbackTexture;

    State(const Device& info, const Target& target)
        : physicalDevice(info.physicalDevice)
        , device(info.device)
        , memoryProperties(info.memoryProperties)
        , target(target)
        , frames(std::max<uint32_t>(info.framesInFlight, 1))
    {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        uniformStride    = alignUp(sizeof(FragUniforms), std::max<VkDeviceSize>(properties.limits.minUniformBufferOffsetAlignment, 16));
        stagingAlignment = std::max<VkDeviceSize>(properties.limits.optimalBufferCopyOffsetAlignment, 16);

        createLayouts();
        vertexShader    = createShader(std::span(nanovg_vk_fill_vert));
        fragmentShader  = createShader(std::span(nanovg_vk_fill_frag));
        fallbackTexture = createOwnedTexture(NVG_TEXTURE_RGBA, 1, 1, 0, nullptr);
        if (!fallbackTexture)
            brls::Logger::error("nanovg: failed to create the fallback texture");
    }

    ~State()
    {
        textures.clear();
        fallbackTexture.reset();
        for (Frame& frame : frames)
        {
            frame.graveyard.clear();
            frame.uniformSet.reset();
        }
    }

    State(const State&)            = delete;
    State& operator=(const State&) = delete;

    static VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment)
    {
        return (value + alignment - 1) / alignment * alignment;
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

    std::optional<Memory> allocateMemory(const VkMemoryRequirements& requirements, VkMemoryPropertyFlags preferred, VkMemoryPropertyFlags required) const
    {
        auto type = findMemoryType(requirements.memoryTypeBits, preferred);
        if (!type)
            type = findMemoryType(requirements.memoryTypeBits, required);
        if (!type)
            return std::nullopt;

        const VkMemoryAllocateInfo info {
            .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize  = requirements.size,
            .memoryTypeIndex = *type,
        };
        VkDeviceMemory memory = VK_NULL_HANDLE;
        if (vkAllocateMemory(device, &info, nullptr, &memory) != VK_SUCCESS)
            return std::nullopt;
        return Memory(device, memory);
    }

    std::optional<HostBuffer> createHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage) const
    {
        const VkBufferCreateInfo info {
            .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size        = size,
            .usage       = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        VkBuffer handle = VK_NULL_HANDLE;
        if (vkCreateBuffer(device, &info, nullptr, &handle) != VK_SUCCESS)
            return std::nullopt;
        Buffer buffer(device, handle);

        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, handle, &requirements);
        constexpr VkMemoryPropertyFlags hostFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        auto memory                               = allocateMemory(requirements, hostFlags, hostFlags);
        if (!memory || vkBindBufferMemory(device, handle, memory->get(), 0) != VK_SUCCESS)
            return std::nullopt;

        void* mapped = nullptr;
        if (vkMapMemory(device, memory->get(), 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)
            return std::nullopt;
        return HostBuffer(std::move(buffer), std::move(*memory), static_cast<std::byte*>(mapped), size);
    }

    void createLayouts()
    {
        const VkDescriptorSetLayoutBinding uniformBinding {
            .binding         = 0,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        };
        const VkDescriptorSetLayoutBinding textureBinding {
            .binding         = 0,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        };
        uniformLayout = createSetLayout(uniformBinding);
        textureLayout = createSetLayout(textureBinding);

        const std::array setLayouts { uniformLayout.get(), textureLayout.get() };
        const VkPushConstantRange pushRange {
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset     = 0,
            .size       = sizeof(float) * 2,
        };
        const VkPipelineLayoutCreateInfo info {
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = static_cast<uint32_t>(setLayouts.size()),
            .pSetLayouts            = setLayouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pushRange,
        };
        VkPipelineLayout layout = VK_NULL_HANDLE;
        if (vkCreatePipelineLayout(device, &info, nullptr, &layout) != VK_SUCCESS)
            brls::Logger::error("nanovg: vkCreatePipelineLayout failed");
        pipelineLayout = PipelineLayout(device, layout);
    }

    DescriptorSetLayout createSetLayout(const VkDescriptorSetLayoutBinding& binding) const
    {
        const VkDescriptorSetLayoutCreateInfo info {
            .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings    = &binding,
        };
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        if (vkCreateDescriptorSetLayout(device, &info, nullptr, &layout) != VK_SUCCESS)
            brls::Logger::error("nanovg: vkCreateDescriptorSetLayout failed");
        return DescriptorSetLayout(device, layout);
    }

    ShaderModule createShader(std::span<const uint32_t> code) const
    {
        const VkShaderModuleCreateInfo info {
            .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = code.size_bytes(),
            .pCode    = code.data(),
        };
        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device, &info, nullptr, &module) != VK_SUCCESS)
            brls::Logger::error("nanovg: vkCreateShaderModule failed");
        return ShaderModule(device, module);
    }

    VkPipeline pipeline(const PipelineKey& key)
    {
        if (auto found = pipelines.find(key); found != pipelines.end())
            return found->second.get();

        const VkSpecializationMapEntry entry { 0, 0, sizeof(VkBool32) };
        const VkBool32 antialias = key.antialias ? VK_TRUE : VK_FALSE;
        const VkSpecializationInfo specialization { 1, &entry, sizeof(VkBool32), &antialias };

        const std::array stages {
            VkPipelineShaderStageCreateInfo {
                .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage  = VK_SHADER_STAGE_VERTEX_BIT,
                .module = vertexShader.get(),
                .pName  = "main",
            },
            VkPipelineShaderStageCreateInfo {
                .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage               = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module              = fragmentShader.get(),
                .pName               = "main",
                .pSpecializationInfo = &specialization,
            },
        };

        const VkVertexInputBindingDescription binding { 0, sizeof(NVGvertex), VK_VERTEX_INPUT_RATE_VERTEX };
        const std::array attributes {
            VkVertexInputAttributeDescription { 0, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(NVGvertex, x)) },
            VkVertexInputAttributeDescription { 1, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(NVGvertex, u)) },
        };
        const VkPipelineVertexInputStateCreateInfo vertexInput {
            .sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount   = 1,
            .pVertexBindingDescriptions      = &binding,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size()),
            .pVertexAttributeDescriptions    = attributes.data(),
        };
        const VkPipelineInputAssemblyStateCreateInfo inputAssembly {
            .sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };
        const VkPipelineViewportStateCreateInfo viewport {
            .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount  = 1,
        };
        const VkPipelineRasterizationStateCreateInfo rasterization {
            .sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode    = VK_CULL_MODE_BACK_BIT,
            .frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .lineWidth   = 1.0f,
        };
        const VkPipelineMultisampleStateCreateInfo multisample {
            .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        };
        const VkStencilOpState keep { VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS, 0xff, 0xff, 0 };
        const VkPipelineDepthStencilStateCreateInfo depthStencil {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .front = keep,
            .back  = keep,
        };
        const VkPipelineColorBlendAttachmentState attachment {
            .blendEnable         = key.colorWrite ? VK_TRUE : VK_FALSE,
            .srcColorBlendFactor = key.blend.srcRGB,
            .dstColorBlendFactor = key.blend.dstRGB,
            .colorBlendOp        = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = key.blend.srcAlpha,
            .dstAlphaBlendFactor = key.blend.dstAlpha,
            .alphaBlendOp        = VK_BLEND_OP_ADD,
            .colorWriteMask      = key.colorWrite ? VkColorComponentFlags(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT) : VkColorComponentFlags(0),
        };
        const VkPipelineColorBlendStateCreateInfo colorBlend {
            .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments    = &attachment,
        };
        constexpr std::array dynamicStates {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
            VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
            VK_DYNAMIC_STATE_STENCIL_REFERENCE,
            VK_DYNAMIC_STATE_CULL_MODE,
            VK_DYNAMIC_STATE_FRONT_FACE,
            VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
            VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
            VK_DYNAMIC_STATE_STENCIL_OP,
        };
        const VkPipelineDynamicStateCreateInfo dynamic {
            .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates    = dynamicStates.data(),
        };
        const VkPipelineRenderingCreateInfo rendering {
            .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount    = 1,
            .pColorAttachmentFormats = &target.colorFormat,
            .depthAttachmentFormat   = VK_FORMAT_UNDEFINED,
            .stencilAttachmentFormat = target.stencilFormat,
        };
        const VkGraphicsPipelineCreateInfo info {
            .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext               = &rendering,
            .stageCount          = static_cast<uint32_t>(stages.size()),
            .pStages             = stages.data(),
            .pVertexInputState   = &vertexInput,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState      = &viewport,
            .pRasterizationState = &rasterization,
            .pMultisampleState   = &multisample,
            .pDepthStencilState  = &depthStencil,
            .pColorBlendState    = &colorBlend,
            .pDynamicState       = &dynamic,
            .layout              = pipelineLayout.get(),
        };

        VkPipeline created = VK_NULL_HANDLE;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &created) != VK_SUCCESS)
        {
            brls::Logger::error("nanovg: vkCreateGraphicsPipelines failed");
            return VK_NULL_HANDLE;
        }
        return pipelines.emplace(key, Pipeline(device, created)).first->second.get();
    }

    VkSampler sampler(int imageFlags)
    {
        const bool mipmaps   = imageFlags & NVG_IMAGE_GENERATE_MIPMAPS;
        const bool nearest   = imageFlags & NVG_IMAGE_NEAREST;
        const bool repeatX   = imageFlags & NVG_IMAGE_REPEATX;
        const bool repeatY   = imageFlags & NVG_IMAGE_REPEATY;
        const uint32_t index = (mipmaps ? 1u : 0u) | (nearest ? 2u : 0u) | (repeatX ? 4u : 0u) | (repeatY ? 8u : 0u);
        if (samplers[index].get() != VK_NULL_HANDLE)
            return samplers[index].get();

        const VkFilter filter = nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        const VkSamplerCreateInfo info {
            .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter    = filter,
            .minFilter    = filter,
            .mipmapMode   = nearest ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = repeatX ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = repeatY ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .minLod       = 0.0f,
            .maxLod       = mipmaps ? VK_LOD_CLAMP_NONE : 0.0f,
        };
        VkSampler created = VK_NULL_HANDLE;
        if (vkCreateSampler(device, &info, nullptr, &created) != VK_SUCCESS)
        {
            brls::Logger::error("nanovg: vkCreateSampler failed");
            return VK_NULL_HANDLE;
        }
        samplers[index] = Sampler(device, created);
        return created;
    }

    std::optional<DescriptorSet> allocateDescriptorSet(VkDescriptorSetLayout layout)
    {
        const VkDescriptorSetAllocateInfo info {
            .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorSetCount = 1,
            .pSetLayouts        = &layout,
        };
        auto tryPool = [&](VkDescriptorPool pool) -> std::optional<DescriptorSet>
        {
            VkDescriptorSetAllocateInfo attempt = info;
            attempt.descriptorPool              = pool;
            VkDescriptorSet set                 = VK_NULL_HANDLE;
            if (vkAllocateDescriptorSets(device, &attempt, &set) != VK_SUCCESS)
                return std::nullopt;
            return DescriptorSet(device, pool, set);
        };

        for (const DescriptorPool& pool : descriptorPools | std::views::reverse)
            if (auto set = tryPool(pool.get()))
                return set;

        const std::array sizes {
            VkDescriptorPoolSize { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, SetsPerPool },
            VkDescriptorPoolSize { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, SetsPerPool },
        };
        const VkDescriptorPoolCreateInfo poolInfo {
            .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets       = SetsPerPool,
            .poolSizeCount = static_cast<uint32_t>(sizes.size()),
            .pPoolSizes    = sizes.data(),
        };
        VkDescriptorPool pool = VK_NULL_HANDLE;
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool) != VK_SUCCESS)
        {
            brls::Logger::error("nanovg: vkCreateDescriptorPool failed");
            return std::nullopt;
        }
        descriptorPools.emplace_back(device, pool);
        return tryPool(pool);
    }

    std::optional<DescriptorSet> createTextureSet(VkImageView view, int imageFlags)
    {
        VkSampler textureSampler = sampler(imageFlags);
        if (textureSampler == VK_NULL_HANDLE)
            return std::nullopt;
        auto set = allocateDescriptorSet(textureLayout.get());
        if (!set)
            return std::nullopt;

        const VkDescriptorImageInfo image {
            .sampler     = textureSampler,
            .imageView   = view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkWriteDescriptorSet write {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = set->get(),
            .dstBinding      = 0,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo      = &image,
        };
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        return set;
    }

    bool supportsMipmaps(VkFormat format) const
    {
        VkFormatProperties properties;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties);
        constexpr VkFormatFeatureFlags required = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        return (properties.optimalTilingFeatures & required) == required;
    }

    std::unique_ptr<Texture> createOwnedTexture(int type, int w, int h, int imageFlags, const unsigned char* data)
    {
        if (w <= 0 || h <= 0)
            return nullptr;

        const VkFormat format = type == NVG_TEXTURE_RGBA ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R8_UNORM;
        uint32_t mipLevels    = 1;
        if ((imageFlags & NVG_IMAGE_GENERATE_MIPMAPS) && supportsMipmaps(format))
            mipLevels = static_cast<uint32_t>(std::bit_width(static_cast<uint32_t>(std::max(w, h))));

        const VkImageCreateInfo imageInfo {
            .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType     = VK_IMAGE_TYPE_2D,
            .format        = format,
            .extent        = { static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1 },
            .mipLevels     = mipLevels,
            .arrayLayers   = 1,
            .samples       = VK_SAMPLE_COUNT_1_BIT,
            .tiling        = VK_IMAGE_TILING_OPTIMAL,
            .usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | (mipLevels > 1 ? VkImageUsageFlags(VK_IMAGE_USAGE_TRANSFER_SRC_BIT) : VkImageUsageFlags(0)),
            .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        VkImage imageHandle = VK_NULL_HANDLE;
        if (vkCreateImage(device, &imageInfo, nullptr, &imageHandle) != VK_SUCCESS)
            return nullptr;
        Image image(device, imageHandle);

        VkMemoryRequirements requirements;
        vkGetImageMemoryRequirements(device, imageHandle, &requirements);
        auto memory = allocateMemory(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        if (!memory || vkBindImageMemory(device, imageHandle, memory->get(), 0) != VK_SUCCESS)
            return nullptr;

        const VkImageViewCreateInfo viewInfo {
            .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image            = imageHandle,
            .viewType         = VK_IMAGE_VIEW_TYPE_2D,
            .format           = format,
            .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1 },
        };
        VkImageView viewHandle = VK_NULL_HANDLE;
        if (vkCreateImageView(device, &viewInfo, nullptr, &viewHandle) != VK_SUCCESS)
            return nullptr;
        ImageView view(device, viewHandle);

        auto set = createTextureSet(viewHandle, imageFlags);
        if (!set)
            return nullptr;

        auto texture = std::make_unique<Texture>(Texture::Description { w, h, type, imageFlags }, std::move(image), std::move(*memory), std::move(view), std::move(*set), mipLevels);
        texture->queueUpload(0, 0, w, h, data);
        return texture;
    }

    int registerTexture(std::unique_ptr<Texture> texture)
    {
        if (!texture)
            return 0;
        const int id = nextTextureId++;
        textures.emplace(id, std::move(texture));
        return id;
    }

    int createTexture(int type, int w, int h, int imageFlags, const unsigned char* data)
    {
        return registerTexture(createOwnedTexture(type, w, h, imageFlags, data));
    }

    int createExternalTexture(VkImageView view, int w, int h, int imageFlags)
    {
        if (view == VK_NULL_HANDLE || w <= 0 || h <= 0)
            return 0;
        auto set = createTextureSet(view, imageFlags);
        if (!set)
            return 0;
        return registerTexture(std::make_unique<Texture>(Texture::Description { w, h, NVG_TEXTURE_RGBA, imageFlags }, std::move(*set)));
    }

    Texture* findTexture(int id) const
    {
        auto found = textures.find(id);
        return found == textures.end() ? nullptr : found->second.get();
    }

    bool deleteTexture(int id)
    {
        auto found = textures.find(id);
        if (found == textures.end())
            return false;
        frames[slot].graveyard.textures.push_back(std::move(found->second));
        textures.erase(found);
        return true;
    }

    bool updateTexture(int id, int x, int y, int w, int h, const unsigned char* data)
    {
        Texture* texture = findTexture(id);
        if (!texture || texture->isExternal())
            return false;
        texture->queueUpload(x, y, w, h, data);
        return true;
    }

    VkDescriptorSet textureSet(int image) const
    {
        if (const Texture* texture = findTexture(image))
            return texture->descriptorSet();
        return fallbackTexture ? fallbackTexture->descriptorSet() : VK_NULL_HANDLE;
    }

    std::optional<Allocation> stage(VkDeviceSize size)
    {
        Frame& frame = frames[slot];
        return frame.staging.allocate(*this, frame.graveyard, size, stagingAlignment);
    }

    void recordUploads()
    {
        if (fallbackTexture)
            fallbackTexture->recordUploads(upload, *this);
        for (auto& [id, texture] : textures)
            texture->recordUploads(upload, *this);
    }

    std::optional<VkDescriptorSet> uniformSetFor(Frame& frame, VkBuffer buffer)
    {
        if (frame.uniformSet && frame.uniformSetBuffer == buffer)
            return frame.uniformSet->get();

        auto set = allocateDescriptorSet(uniformLayout.get());
        if (!set)
            return std::nullopt;

        const VkDescriptorBufferInfo bufferInfo { buffer, 0, sizeof(FragUniforms) };
        const VkWriteDescriptorSet write {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = set->get(),
            .dstBinding      = 0,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .pBufferInfo     = &bufferInfo,
        };
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

        if (frame.uniformSet)
            frame.graveyard.sets.push_back(std::move(*frame.uniformSet));
        frame.uniformSet       = std::move(set);
        frame.uniformSetBuffer = buffer;
        return frame.uniformSet->get();
    }

    void flush(const Context& context)
    {
        if (!frameActive)
            return;

        recordUploads();

        auto calls        = context.recordedCalls();
        auto vertices     = context.recordedVertices();
        auto uniforms     = context.recordedUniforms();
        const auto view   = context.viewSize();
        const float ratio = context.devicePixelRatio();
        if (drawCmd == VK_NULL_HANDLE || calls.empty() || vertices.empty() || view[0] <= 0.0f || view[1] <= 0.0f)
            return;

        Frame& frame      = frames[slot];
        auto vertexSlice  = frame.vertices.allocate(*this, frame.graveyard, vertices.size_bytes(), sizeof(NVGvertex));
        auto uniformSlice = frame.uniforms.allocate(*this, frame.graveyard, uniforms.size() * uniformStride, uniformStride);
        if (!vertexSlice || !uniformSlice)
        {
            brls::Logger::error("nanovg: failed to allocate frame buffers");
            return;
        }
        auto uniformSet = uniformSetFor(frame, uniformSlice->buffer);
        if (!uniformSet)
            return;

        std::memcpy(vertexSlice->data, vertices.data(), vertices.size_bytes());
        for (size_t index : std::views::iota(size_t { 0 }, uniforms.size()))
            std::memcpy(uniformSlice->data + index * uniformStride, &uniforms[index], sizeof(FragUniforms));

        const VkViewport viewport { 0.0f, 0.0f, view[0] * ratio, view[1] * ratio, 0.0f, 1.0f };
        const VkRect2D scissor { { 0, 0 }, { static_cast<uint32_t>(std::lround(view[0] * ratio)), static_cast<uint32_t>(std::lround(view[1] * ratio)) } };

        VkCommandBuffer cmd = drawCmd;
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdSetFrontFace(cmd, VK_FRONT_FACE_COUNTER_CLOCKWISE);
        vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
        vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        vkCmdSetStencilTestEnable(cmd, VK_FALSE);
        vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, 0xff);
        vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, 0xff);
        vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, 0);
        vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);

        const VkBuffer vertexBuffer     = vertexSlice->buffer;
        const VkDeviceSize vertexOffset = vertexSlice->offset;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &vertexOffset);
        vkCmdPushConstants(cmd, pipelineLayout.get(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float) * 2, view.data());

        Recorder recorder(*this, context, cmd, *uniformSet, uniformSlice->offset);
        for (const Call& call : calls)
            recorder.record(call);
    }

    void setFrame(VkCommandBuffer uploadBuffer, VkCommandBuffer drawBuffer, uint32_t frameSlot)
    {
        if (uploadBuffer == VK_NULL_HANDLE)
        {
            frameActive = false;
            upload      = VK_NULL_HANDLE;
            drawCmd     = VK_NULL_HANDLE;
            return;
        }

        slot        = frameSlot % static_cast<uint32_t>(frames.size());
        upload      = uploadBuffer;
        drawCmd     = drawBuffer;
        frameActive = true;

        Frame& frame = frames[slot];
        frame.graveyard.clear();
        frame.vertices.reset();
        frame.uniforms.reset();
        frame.staging.reset();
    }
};

VkRenderer::VkRenderer(const Device& device, const Target& target)
    : state(std::make_unique<State>(device, target))
{
}

VkRenderer::~VkRenderer() = default;

void VkRenderer::setFrame(VkCommandBuffer upload, VkCommandBuffer draw, uint32_t slot)
{
    state->setFrame(upload, draw, slot);
}

int VkRenderer::createImageFromView(NVGcontext*, VkImageView view, int width, int height, int imageFlags)
{
    return state->createExternalTexture(view, width, height, imageFlags);
}

NVGcontext* VkRenderer::createContext(int flags)
{
    auto* context    = new State::Context(*state, flags);
    NVGparams params = State::Context::params(context);
    return nvgCreateInternal(&params);
}

void VkRenderer::deleteContext(NVGcontext* ctx)
{
    if (ctx)
        nvgDeleteInternal(ctx);
}

} // namespace nvg
