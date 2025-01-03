add_library(websocket STATIC
  websocket_client.cc
  websocket_server.cc
)
target_link_libraries(websocket PUBLIC decoder)
# 第二顶层的代码，第一顶层是bin里的main函数，第三是decoder，frontend