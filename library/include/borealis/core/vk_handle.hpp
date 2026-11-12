#pragma once

#include <vulkan/vulkan.h>

#include <utility>

namespace brls::vk
{

template <typename Handle, auto Destroy>
class RootHandle
{
  public:
    RootHandle() = default;
    explicit RootHandle(Handle handle)
        : handle(handle)
    {
    }
    ~RootHandle() { reset(); }
    RootHandle(const RootHandle&)            = delete;
    RootHandle& operator=(const RootHandle&) = delete;
    RootHandle(RootHandle&& other) noexcept
        : handle(std::exchange(other.handle, VK_NULL_HANDLE))
    {
    }
    RootHandle& operator=(RootHandle&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            handle = std::exchange(other.handle, VK_NULL_HANDLE);
        }
        return *this;
    }

    Handle get() const { return handle; }
    explicit operator bool() const { return handle != VK_NULL_HANDLE; }

    void reset()
    {
        if (handle != VK_NULL_HANDLE)
            Destroy(std::exchange(handle, VK_NULL_HANDLE), nullptr);
    }

  private:
    Handle handle = VK_NULL_HANDLE;
};

template <typename Parent, typename Handle, auto Destroy>
class ChildHandle
{
  public:
    ChildHandle() = default;
    ChildHandle(Parent parent, Handle handle)
        : parent(parent)
        , handle(handle)
    {
    }
    ~ChildHandle() { reset(); }
    ChildHandle(const ChildHandle&)            = delete;
    ChildHandle& operator=(const ChildHandle&) = delete;
    ChildHandle(ChildHandle&& other) noexcept
        : parent(other.parent)
        , handle(std::exchange(other.handle, VK_NULL_HANDLE))
    {
    }
    ChildHandle& operator=(ChildHandle&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            parent = other.parent;
            handle = std::exchange(other.handle, VK_NULL_HANDLE);
        }
        return *this;
    }

    Handle get() const { return handle; }
    explicit operator bool() const { return handle != VK_NULL_HANDLE; }

    void reset()
    {
        if (handle != VK_NULL_HANDLE)
            Destroy(parent, std::exchange(handle, VK_NULL_HANDLE), nullptr);
    }

  private:
    Parent parent = VK_NULL_HANDLE;
    Handle handle = VK_NULL_HANDLE;
};

template <typename Handle, auto Destroy>
using DeviceHandle = ChildHandle<VkDevice, Handle, Destroy>;

} // namespace brls::vk
