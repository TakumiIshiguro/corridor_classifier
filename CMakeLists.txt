cmake_minimum_required(VERSION 3.0.2)
project(corridor_classifier)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  message_filters
  rospy
  scenario_navigation_msgs
  sensor_msgs
  std_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    cv_bridge
    message_filters
    rospy
    scenario_navigation_msgs
    sensor_msgs
    std_msgs
)

catkin_install_python(PROGRAMS
  scripts/add_depth_to_dataset.py
  scripts/add_bev_to_dataset.py
  scripts/create_dataset.py
  scripts/corridor_classifier_node.py
  scripts/evaluate_session_holdout.py
  scripts/fit_production_linear_probe.py
  scripts/linear_probe_node.py
  scripts/relabel_turning.py
  scripts/train.py
  scripts/train_linear_probe.py
  scripts/tune_passage_thresholds.py
  scripts/visualize_features.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  config
  launch
  weights
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES
  LICENSE
  README.md
  THIRD_PARTY_NOTICES.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
