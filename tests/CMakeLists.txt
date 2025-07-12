# tests/CMakeLists.txt

include(${CMAKE_SOURCE_DIR}/cmake/ProjectConfig.cmake)


# להגדיר קבצי הטסטים שלך
# set(TEST_SOURCES
#     triangle_cpp_test.cpp
#     utils_cpp_test.cpp
#     ...
# )

# add_executable(run_all_tests ${TEST_SOURCES})
# # לקשר לספריות שאתה בודק (שנבנות בפרויקט הראשי)
# target_link_libraries(run_all_tests PRIVATE triangle_detector_cpp_for_test ${OpenCV_LIBS} gtest gtest_main)

# # תוספות אופציונליות
# gtest_discover_tests(run_all_tests)