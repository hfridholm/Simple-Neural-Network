
#include "../Header-Files-Folder/engine-include-header.h"

bool train_network_stcast(int layerAmount, const int layerSizes[], const int layerActivs[], float*** weights, float** biases, float learnRate, float momentum, float* inputs, float* targets)
{
  int maxShape = maximum_layer_shape(layerSizes, layerAmount);

  float*** weightDeltas = create_fmatrix_array(layerAmount - 1, maxShape, maxShape);
  float** biasDeltas = create_float_matrix(layerAmount - 1, maxShape);


  // Calculate derivatives
  float*** weightDerivs = create_fmatrix_array(layerAmount - 1, maxShape, maxShape);
  float** biasDerivs = create_float_matrix(layerAmount - 1, maxShape);


  float** nodeValues = create_float_matrix(layerAmount, maxShape);

  create_node_values(nodeValues, layerAmount, layerSizes, layerActivs, weights, biases, inputs);
  create_weibia_derivs(weightDerivs, biasDerivs, layerAmount, layerSizes, layerActivs, weights, biases, nodeValues, targets);

  free_float_matrix(nodeValues, layerAmount, maxShape);


  // Calculate deltas
  create_weibia_deltas(weightDeltas, biasDeltas, layerAmount, layerSizes, learnRate, momentum, weightDerivs, biasDerivs, NULL, NULL);

  free_fmatrix_array(weightDerivs, layerAmount - 1, maxShape, maxShape);
  free_float_matrix(biasDerivs, layerAmount - 1, maxShape);


  // Update weights and biases using deltas
  addit_elem_fmatarr(weights, weights, weightDeltas, layerAmount - 1, maxShape, maxShape);

  addit_elem_fmatrix(biases, biases, biasDeltas, layerAmount - 1, maxShape);


  free_fmatrix_array(weightDeltas, layerAmount - 1, maxShape, maxShape);
  free_float_matrix(biasDeltas, layerAmount - 1, maxShape);


  return true;
}

bool train_network_minbat(int layerAmount, const int layerSizes[], const int layerActivs[], float*** weights, float** biases, float learnRate, float momentum, float** inputs, float** targets, int batchSize)
{
  int maxShape = maximum_layer_shape(layerSizes, layerAmount);

  float*** weightDeltas = create_fmatrix_array(layerAmount - 1, maxShape, maxShape);
  float** biasDeltas = create_float_matrix(layerAmount - 1, maxShape);


  float*** meanWeightDerivs = create_fmatrix_array(layerAmount - 1, maxShape, maxShape);
  float** meanBiasDerivs = create_float_matrix(layerAmount - 1, maxShape);


  float*** weightDerivs = create_fmatrix_array(layerAmount - 1, maxShape, maxShape);
  float** biasDerivs = create_float_matrix(layerAmount - 1, maxShape);

  for(int inputIndex = 0; inputIndex < batchSize; inputIndex += 1)
  {
    // Calculate derivatives
    float** nodeValues = create_float_matrix(layerAmount, maxShape);

    create_node_values(nodeValues, layerAmount, layerSizes, layerActivs, weights, biases, inputs[inputIndex]);
    create_weibia_derivs(weightDerivs, biasDerivs, layerAmount, layerSizes, layerActivs, weights, biases, nodeValues, targets[inputIndex]);

    free_float_matrix(nodeValues, layerAmount, maxShape);

    addit_elem_fmatarr(meanWeightDerivs, meanWeightDerivs, weightDerivs, layerAmount - 1, maxShape, maxShape);
    addit_elem_fmatrix(meanBiasDerivs, meanBiasDerivs, biasDerivs, layerAmount - 1, maxShape);

  }

  free_fmatrix_array(weightDerivs, layerAmount - 1, maxShape, maxShape);
  free_float_matrix(biasDerivs, layerAmount - 1, maxShape);


  multi_scale_fmatarr(meanWeightDerivs, meanWeightDerivs, layerAmount - 1, maxShape, maxShape, 1.0f / batchSize);
  multi_scale_fmatrix(meanBiasDerivs, meanBiasDerivs, layerAmount - 1, maxShape, 1.0f / batchSize);


  // Calculate deltas
  create_weibia_deltas(weightDeltas, biasDeltas, layerAmount, layerSizes, learnRate, momentum, meanWeightDerivs, meanBiasDerivs, NULL, NULL);


  free_fmatrix_array(meanWeightDerivs, layerAmount - 1, maxShape, maxShape);
  free_float_matrix(meanBiasDerivs, layerAmount - 1, maxShape);


  // Update weights and biases using deltas
  addit_elem_fmatarr(weights, weights, weightDeltas, layerAmount - 1, maxShape, maxShape);
  addit_elem_fmatrix(biases, biases, biasDeltas, layerAmount - 1, maxShape);


  free_fmatrix_array(weightDeltas, layerAmount - 1, maxShape, maxShape);
  free_float_matrix(biasDeltas, layerAmount - 1, maxShape);


  return true;
}

bool frwrd_network_inputs(float* outputs, int layerAmount, const int layerSizes[], const int layerActivs[], float*** weights, float** biases, float* inputs)
{
  int maxShape = maximum_layer_shape(layerSizes, layerAmount);

  float* layerInputs = create_float_vector(maxShape);
  float* layerOutputs = create_float_vector(maxShape);

  copy_float_vector(layerInputs, inputs, layerSizes[0]);

  for(int layerIndex = 1; layerIndex < layerAmount; layerIndex += 1)
  {
    int layerHeight = layerSizes[layerIndex], layerWidth = layerSizes[layerIndex - 1];


    dotprod_fmatrix_vector(layerOutputs, weights[layerIndex - 1], layerHeight, layerWidth, layerInputs, layerWidth);

    addit_elem_fvector(layerOutputs, layerOutputs, biases[layerIndex - 1], layerHeight);



    float* (*layer_activat_funct)(float*, float*, int);
    if(parse_activat_funct(&layer_activat_funct, layerActivs[layerIndex - 1]))
    {
      layer_activat_funct(layerOutputs, layerOutputs, layerHeight);
    }
    else printf("Error: parse_active_funct\n");


    copy_float_vector(layerInputs, layerOutputs, maxShape);
  }

  copy_float_vector(outputs, layerOutputs, layerSizes[layerAmount - 1]);

  free_float_vector(layerInputs, maxShape);
  free_float_vector(layerOutputs, maxShape);

  return true;
}
