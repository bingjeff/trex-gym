# T-Rex WASM Static App

This branch is a self-contained static version of the joystick T-Rex web app
for GitHub Pages. It runs MuJoCo physics, policy inference, keyboard/touch
command handling, and Three.js rendering in the browser.

Generated artifacts are produced from the full `wasm` branch with:

```sh
JAX_PLATFORMS=cpu uv run python tools/export_web_wasm_assets.py
```

To test this branch locally:

```sh
python3 -m http.server 8766 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8766/`.
