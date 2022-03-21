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

#include "image_provider.h"
#include "YoloV5Weights.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/c/c_api_types.h"

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

float TP = 0;
float FP = 0;
float FN = 0;
int vorhersagen_gesamt = 0;

int Geradeaus = 0;
int Kreisverkehr = 0;
int Links = 0;
int Rechts = 0;
int Stop = 0;

int j = 0;

// In order to use optimized tensorflow lite kernels, a signed int8_t quantized
// model is preferred over the legacy unsigned model format. This means that
// throughout this project, input images must be converted from unisgned to
// signed format. The easiest and quickest way to convert from unsigned to
// signed 8-bit integers is to subtract 128 from the unsigned value to get a
// signed value.

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 300 * 1024;
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
  model = tflite::GetModel(uc_final_model_tflite);
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

  heap_caps_print_heap_info(MALLOC_CAP_SPIRAM);
  
  // heap_caps_print_heap_info(MALLOC_CAP_32BIT);

  // heap_caps_print_heap_info(MALLOC_CAP_INTERNAL);

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
 
  tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  // static tflite::MicroMutableOpResolver<14> micro_op_resolver;
  //   micro_op_resolver.AddStridedSlice();
  //   micro_op_resolver.AddConcatenation();
  //   micro_op_resolver.AddMul();
  //   micro_op_resolver.AddConv2D();
  //   micro_op_resolver.AddLogistic();
  //   micro_op_resolver.AddPad();
  //   micro_op_resolver.AddAdd();
  //   micro_op_resolver.AddMaxPool2D();
  //   micro_op_resolver.AddQuantize();
  //   micro_op_resolver.AddResizeNearestNeighbor();
  //   micro_op_resolver.AddReshape();
  //   micro_op_resolver.AddTranspose();
  //   micro_op_resolver.AddSub();
  //   micro_op_resolver.AddDequantize();



  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, 
      kTensorArenaSize, error_reporter);
      interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
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

  // printf("Dimension Input Tensor = %d\n", input->dims->size);
  // printf("input size in bytes = %d\n", input->bytes);
  // printf("Datatype? = %d\n", input->type);
  // printf("Batch = %d\n",input->dims->data[0]);
  // printf("Width = %d\n",input->dims->data[1]);
  // printf("High = %d\n",input->dims->data[2]);
  // printf("Channel = %d\n\n",input->dims->data[3]);

  // printf("Dimension Output Tensor = %d\n", output->dims->size);
  // printf("output size in bytes = %d\n", output->bytes);
  // printf("Datatype? = %d\n", output->type);
  // printf("Rank one = %d\n",output->dims->data[0]);
  // printf("Rank two = %d\n",output->dims->data[1]);
  // printf("Rank three = %d\n",output->dims->data[2]);
  // printf("Rank four = %d\n\n",output->dims->data[3]);

const int nc = 5;
int num_output = 1;
int index = 0;

float confidence_score = 0;
float class_conf[nc];
float class_prob[nc];
float max_class_conf = 0;
int max_class_conf_index = 5;

// Größe des Arrays auslesen
for (int i = 0; i < output->dims->size; i++)
{
  num_output = num_output * output->dims->data[i];
}

// printf("Größe des Arrays = %d\n\n", num_output);

// Die Ausgabe nach dem höchsten Confidence Score absuchen
// und den Index des höchsten Wertes zum übergeben speichern
for(int i = 4; i <= num_output; i = i + 10)
{
  if(output->data.f[i] > confidence_score )
  {
    confidence_score = output->data.f[i];
    index = i;
  }
}

// printf("Index : %d | confidence_score: %f\n\n", index, confidence_score);

// Die zum höchsten Vertrauenswert gehörenden Klassenwahrscheinlichkeiten
// übergeben
for (int i = 0 ; i < nc; i++)
{
  class_prob[i] = output->data.f[index + i + 1];
  // printf("Klasse = %d | class_prob = %.2f\n", i, class_prob[i]);
}

// printf("\n");
// Berechnen und Ausgeben der Klassenvertrauenswerte
for (int i = 0; i < nc; i++)
{
  class_conf[i] = confidence_score * class_prob[i];
  // printf("Klasse = %d | class_conf = %.2f\n", i, class_conf[i]);
}

// printf("\n");

// suchen nach dem höchsten Klassenvertrauenswert
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
