/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

#include <esp_heap_caps.h>

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 1500 * 1024;
static uint8_t *tensor_arena;//[kTensorArenaSize]; // Maybe we should move this to external
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(best_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  if (tensor_arena == NULL) {
    tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  // static tflite::MicroMutableOpResolver<8> micro_op_resolver;
  // micro_op_resolver.AddResizeNearestNeighbor();
  // micro_op_resolver.AddMul();
  // micro_op_resolver.AddSub();
  // micro_op_resolver.AddConv2D();
  // micro_op_resolver.AddDepthwiseConv2D();
  // micro_op_resolver.AddPad();
  // micro_op_resolver.AddAveragePool2D();
  // micro_op_resolver.AddLogistic();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
}

// The name of this function is important for Arduino compatibility.
void loop() {
  float start = clock();

  // Get image from provider.
  if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                            input->data.int8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
  }
  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  printf("input size = %d\n", input->dims->size);
  printf("input size in bytes = %d\n", input->bytes);
  printf("Is input int8? = %d\n", input->type);
  printf("Batch = %d\n",input->dims->data[0]);
  printf("Width = %d\n",input->dims->data[1]);
  printf("High = %d\n",input->dims->data[2]);
  printf("Channel = %d\n\n",input->dims->data[3]);

  printf("output size = %d\n", output->dims->size);
  printf("output size in bytes = %d\n", output->bytes);
  printf("Is output int8? = %d\n", output->type);
  printf("Batch = %d\n",output->dims->data[0]);
  printf("Width = %d\n",output->dims->data[1]);
  printf("High = %d\n",output->dims->data[2]);
  printf("Channel = %d\n\n",output->dims->data[3]);

  // Process the inference results.
  // int8_t geradeaus = output->data.uint8[0];
  // int8_t kreisverkehr = output->data.uint8[1];
  // int8_t links = output->data.uint8[2];
  // int8_t negative = output->data.uint8[3];
  // int8_t rechts = output->data.uint8[4];
  // int8_t stoppschild = output->data.uint8[5];
  
  // printf("geradeaus = %d\n", geradeaus);
  // printf("kreisverkehr = %d\n", kreisverkehr);
  // printf("links = %d\n", links);
  // printf("negative = %d\n", negative);
  // printf("rechts = %d\n", rechts);
  // printf("stoppschild = %d\n", stoppschild);
  // RespondToDetection(error_reporter, stop_score, negative_score, car_score);

    float conf[4][2];

    conf[0][0] = 0;
    conf[1][0] = 1;
    conf[2][0] = 2;
    conf[3][0] = 3;
    conf[4][0] = 4;

    int num_output = output->dims->data[1] * output->dims->data[2];

        float conf0_max = 0;
        float conf1_max = 0;
        float conf2_max = 0;
        float conf3_max = 0;
        float conf4_max = 0;

    // for(int i = 0; i <= num_output/1; i = i + 10)
    // {

    //   printf("index: %d x: %f\n", i, output->data.f[i]);
    //   printf("index: %d y: %f\n", i+1, output->data.f[i+1]);
    //   printf("index: %d w: %f\n", i+2,output->data.f[i+2]);
    //   printf("index: %d h: %f\n\n", i+3,output->data.f[i+3]);
    //   printf("index: %d conf: %f\n\n", i+4,output->data.f[i+4]);
    //   printf("index: %d class0: %f\n", i+5,output->data.f[i+5]);
    //   printf("index: %d class1: %f\n", i+6,output->data.f[i+6]);
    //   printf("index: %d class2: %f\n", i+7,output->data.f[i+7]);
    //   printf("index: %d class3: %f\n", i+8,output->data.f[i+8]);
    //   printf("index: %d class4: %f\n\n", i+9,output->data.f[i+9]);
    // }



    for(int i = 4; i <= num_output; i=i+10)
    {
   
      float objectness_score = output->data.f[i];
      float class_0_score = output->data.f[i + 1];
      float class_1_score = output->data.f[i + 2];
      float class_2_score = output->data.f[i + 3];
      float class_3_score = output->data.f[i + 4];
      float class_4_score = output->data.f[i + 5];

        // printf("%f\n\n", objectness_score);
        // printf("%f\n", class_0_score);
        // printf("%f\n", class_1_score);
        // printf("%f\n", class_2_score);
        // printf("%f\n", class_3_score);
        // printf("%f\n\n", class_4_score);

      conf[0][1] = objectness_score * class_0_score;
      conf[1][1] = objectness_score * class_1_score;
      conf[2][1] = objectness_score * class_2_score;
      conf[3][1] = objectness_score * class_3_score;
      conf[4][1] = objectness_score * class_4_score;

        // printf("%f\n", conf[0][1]);
        // printf("%f\n", conf[1][1]);
        // printf("%f\n", conf[2][1]);
        // printf("%f\n", conf[3][1]);
        // printf("%f\n\n", conf[4][1]);
        


        if(conf[0][1] > conf0_max )
        {
          conf0_max = conf[0][1];
        }

        if(conf[1][1] > conf1_max )
        {
          conf1_max = conf[1][1];
        }

       if(conf[2][1] > conf2_max )
        {
          conf2_max = conf[2][1];
        }

        if(conf[3][1] > conf3_max )
        {
          conf3_max = conf[3][1];
        }

       if(conf[4][1] > conf4_max )
        {
          conf4_max = conf[4][1];
        }


    }

        printf("class0: %f\n", conf0_max);
        printf("class1: %f\n", conf1_max);
        printf("class2: %f\n", conf2_max);
        printf("class3: %f\n", conf3_max);
        printf("class4: %f\n", conf4_max);


  float end = clock();
  int time = end - start;
  printf("Zeit pro Durchlauf = %d\n", time);

}


      // for(j = 0; j < 5; j++)
      // {
      //   float confidence = conf[j][1];
      //   if( confidence > conf_max)
      //     {
      //       conf_max = conf[j][1];
      //       class_max = conf[j][0];
      //       printf("conf max: %f\n", conf_max);
      //       printf("class_max: %.f\n\n", class_max);
      //     }
      // }