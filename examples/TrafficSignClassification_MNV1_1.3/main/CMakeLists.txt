
#
# Main component of TF Micro project 'person_detection'.
#

idf_component_register(
    SRCS
    "image_provider.cc"
    "main.cc"
    "main_functions_MNV1.cc"      
    "app_camera_esp.c"
    "traffic_sign_model_data_MNV1.cc"
    PRIV_REQUIRES console tflite-lib esp32-camera spi_flash
    INCLUDE_DIRS "")
