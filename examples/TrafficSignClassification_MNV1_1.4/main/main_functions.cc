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

/* @Author:       Marco Lotzkat
 * @LastEditTime: 01.03.2022
 * @LastEditors:  Marco Lotzkat            
 */

#include "main_functions.h"

#include "image_provider.h"
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

int kNumCols;
int kNumRows;
int kNumChannels;

const int nc = 6; 
int j = 0;

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

  printf("%d\n", heap_caps_get_total_size(MALLOC_CAP_SPIRAM));

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.

  tflite::AllOpsResolver micro_op_resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  // static tflite::MicroMutableOpResolver<8> micro_op_resolver;
  // micro_op_resolver.AddResizeNearestNeighbor();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
  input = interpreter->input(0);
  kNumCols = input->dims->data[1];
  kNumRows = input->dims->data[2];
  kNumChannels = input->dims->data[3];
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
  printf("Datatype?? = %d\n", output->type);

  for (int i = 0; i < output->dims->size; i++)
  {
    printf("Rank %d = %d\n", i + 1,output->dims->data[i]);
  }

uint8_t class_conf[nc];
uint8_t max_class_conf = 0; //mit kleinster Wahrscheinlichkeit deklariert
int max_class_conf_index = 6; // mit random Zahl deklariert

// Alle Vorhersagen werden einem Array übergeben
for (int i = 0; i < nc; i++)
{
  class_conf[i] = output->data.uint8[i];
  printf("class_conf = %d ", class_conf[i]);
}

// printf("\n");
// Die höchste Wahrscheinlichkeit + Index wird ermittelt
for (int i = 0; i < nc; i++)
{
  if (class_conf[i] > max_class_conf)
  {
    max_class_conf = class_conf[i];
    max_class_conf_index = i;
  }
}

j += 1;

printf("%d, %d\n", max_class_conf_index, j);

}
