add_library(http STATIC
  http_client.cc
  http_server.cc
)
target_link_libraries(http PUBLIC decoder)
# 第二顶层的代码，第一顶层是bin里的main函数，第三是decoder，frontend