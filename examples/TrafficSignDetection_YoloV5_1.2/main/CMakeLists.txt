
#
# Main component of TF Micro project 'person_detection'.
#

idf_component_register(
    SRCS  
        "image_provider.cc"
        "main.cc"
        "main_functions.cc"
        "YoloV5_Weights.cc"
        "app_camera_esp.c"
    PRIV_REQUIRES console tflite-lib esp32-camera spi_flash
    INCLUDE_DIRS "")
