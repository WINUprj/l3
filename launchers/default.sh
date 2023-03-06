#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# launching app
dt-exec roslaunch led_emitter led_emitter_node.launch veh:=csc22918
dt-exec roslaunch duckietown_demos deadreckoning.launch
# dt-exec roslaunch augmented_reality augmented_reality.launch map_file:="/data/maps/mapfile.yaml" veh:=csc22918
dt-exec roslaunch augmented_reality_apriltag augmented_reality_apriltag.launch
# dt-exec roslaunch lane_following lane_following.launch

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
