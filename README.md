# atom

Meson-based C++ project with a base library and algorithm apps (yolact, siv).

- `base/`: shared components like TensorRT wrapper
- `algorithms/yolact`: example algorithm app depending on base
- `algorithms/siv`: another algorithm app depending on base

Build with Meson:

```sh
meson setup build
meson compile -C build
meson test -C build || true
```
