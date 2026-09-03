#!/bin/bash
# Common container initialization: ROS / Autoware workspace setup.

ros_distro="${ROS_DISTRO:-humble}"

if [ -f /autoware/install/setup.bash ]; then
    # shellcheck disable=SC1091
    set +u && source /autoware/install/setup.bash \
        > >(grep -v 'not found: "/opt/ros/jazzy/local_setup.bash"') \
        2> >(grep -v 'not found: "/opt/ros/jazzy/local_setup.bash"' >&2)
elif [ -f "/opt/ros/${ros_distro}/setup.bash" ]; then
    # shellcheck disable=SC1091
    set +u && source "/opt/ros/${ros_distro}/setup.bash"
elif [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck disable=SC1091
    set +u && source /opt/ros/humble/setup.bash
fi

# --- Source ROS workspace (skip when not yet built, e.g. first dev session) ---
if [ -f /aichallenge/workspace/install/setup.bash ]; then
    # shellcheck disable=SC1091
    set +u && source /aichallenge/workspace/install/setup.bash \
        > >(grep -v 'not found: "/opt/ros/jazzy/local_setup.bash"') \
        2> >(grep -v 'not found: "/opt/ros/jazzy/local_setup.bash"' >&2)
fi

# When used as ENTRYPOINT, hand off to the CMD / command.
# When sourced from .bashrc, exec is a no-op (no positional args).
if [ $# -gt 0 ]; then
    exec "$@"
fi
