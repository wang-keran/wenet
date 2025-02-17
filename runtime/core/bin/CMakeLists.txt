add_executable(decoder_main decoder_main.cc)
target_link_libraries(decoder_main PUBLIC decoder)
if(IPEX)
  target_link_libraries(decoder_main PUBLIC "${TORCH_IPEX_LIBRARIES}")
endif()

add_executable(label_checker_main label_checker_main.cc)
target_link_libraries(label_checker_main PUBLIC decoder)

if(TORCH)
 add_executable(api_main api_main.cc)
 target_link_libraries(api_main PUBLIC wenet_api)
endif()

if(WEBSOCKET)
  add_executable(websocket_client_main websocket_client_main.cc)
  target_link_libraries(websocket_client_main PUBLIC websocket)
  add_executable(websocket_server_main websocket_server_main.cc)
  target_link_libraries(websocket_server_main PUBLIC websocket)
endif()

if(GRPC)
  add_executable(grpc_server_main grpc_server_main.cc)
  target_link_libraries(grpc_server_main PUBLIC wenet_grpc)
  add_executable(grpc_client_main grpc_client_main.cc)
  target_link_libraries(grpc_client_main PUBLIC wenet_grpc)
endif()

if(HTTP)
  add_executable(http_client_main http_client_main.cc)
  target_link_libraries(http_client_main PUBLIC http)
  add_executable(http_server_main http_server_main.cc)
  target_link_libraries(http_server_main PUBLIC http)
endif()

# 判断使用哪种通信形式