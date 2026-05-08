[![CI](https://github.com/distilledecho/kai-daemon/actions/workflows/ci.yml/badge.svg)](https://github.com/distilledecho/kai-daemon/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/distilledecho/kai-daemon/branch/main/graph/badge.svg)](https://codecov.io/gh/distilledecho/kai-daemon)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# kai_daemon

The daemon — persistent memory, background workflows, and inner life pipeline

This is where you should write a short paragraph that describes what your module does,
how it does it, and why people should use it.

Source          | <https://github.com/distilledecho/kai-daemon>
:---:           | :---:
Documentation   | <https://distilledecho.github.io/kai-daemon>
Releases        | <https://github.com/distilledecho/kai-daemon/releases>

This is where you should put some images or code snippets that illustrate
some relevant examples. If it is a library then you might put some
introductory code here:

```python
from kai_daemon import __version__

print(f"Hello kai_daemon {__version__}")
```

Or if it is a commandline tool then you might put some example commands here:

```
python -m kai_daemon --version
```

## Running as a launchd Service

The recommended way to run kai-daemon in production is as a macOS user agent via launchd. This ensures the daemon restarts automatically after a crash.

kai-daemon connects to mlx-kv-server **lazily** — on the first inference call, not at startup — so it can be loaded independently of mlx-kv-server's state.

### Install

```bash
launchctl load ~/dev/kai-daemon/com.distilledecho.kai-daemon.plist
```

This registers the service. The daemon does **not** start immediately (`RunAtLoad` is false) — start it manually after loading:

```bash
launchctl start com.distilledecho.kai-daemon
```

### Stop / Start / Restart

```bash
# Stop (launchd will restart it after ThrottleInterval if KeepAlive is active)
launchctl stop com.distilledecho.kai-daemon

# To stop permanently without restarting, unload first (see Uninstall below)

# Start manually
launchctl start com.distilledecho.kai-daemon
```

### Logs

stdout and stderr are written to separate files:

```bash
tail -f ~/dev/kai-daemon/logs/stdout.log
tail -f ~/dev/kai-daemon/logs/stderr.log
```

### Uninstall

```bash
launchctl unload ~/dev/kai-daemon/com.distilledecho.kai-daemon.plist
```

This stops the daemon and removes launchd's awareness of it. The plist file remains on disk; re-load to re-register.

### Notes

- `ThrottleInterval` is set to 10 seconds. If mlx-kv-server is down, the daemon will crash on its first inference attempt and launchd will restart it after 10 seconds rather than immediately.
- The plist hardcodes the absolute path to `uv` (`/opt/homebrew/bin/uv`). If uv is reinstalled elsewhere, update `ProgramArguments[0]` in the plist and reload.

<!-- README only content. Anything below this line won't be included in index.md -->

See https://distilledecho.github.io/kai-daemon for more detailed documentation.
