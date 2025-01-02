add_library(frontend STATIC
  feature_pipeline.cc
  fft.cc
)
target_link_libraries(frontend PUBLIC utils)
# 引入前段的脚本
# 第二顶层的代码，第一顶层是bin里的main函数，第三是decoder，frontend