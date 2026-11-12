#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>

#include "nanovg.h"

enum NVGcreateFlags
{
    NVG_ANTIALIAS       = 1 << 0,
    NVG_STENCIL_STROKES = 1 << 1,
    NVG_DEBUG           = 1 << 2,
};

namespace nvg
{

class VkRenderer
{
  public:
    struct Device
    {
        VkPhysicalDevice physicalDevice;
        VkDevice device;
        VkPhysicalDeviceMemoryProperties memoryProperties;
        uint32_t framesInFlight;
    };

    struct Target
    {
        VkFormat colorFormat;
        VkFormat stencilFormat;
    };

    VkRenderer(const Device& device, const Target& target);
    ~VkRenderer();
    VkRenderer(const VkRenderer&)            = delete;
    VkRenderer& operator=(const VkRenderer&) = delete;

    void setFrame(VkCommandBuffer upload, VkCommandBuffer draw, uint32_t slot);
    int createImageFromView(NVGcontext* ctx, VkImageView view, int width, int height, int imageFlags);

    NVGcontext* createContext(int flags);
    static void deleteContext(NVGcontext* ctx);

  private:
    struct State;
    std::unique_ptr<State> state;
};

} // namespace nvg
