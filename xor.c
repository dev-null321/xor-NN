#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUTNODES 2
#define HIDDENNODES 2
#define OUTPUTNODES 1

const int epochs_count = 10000;
double learning_rate = 0.1;

double xor_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
double xor_outputs[4] = {0, 1, 1, 0};

void initialize_weights_biases(double weightInputHidden[HIDDENNODES][INPUTNODES],
                               double weightHiddenOutput[OUTPUTNODES][HIDDENNODES],
                               double biasHidden[HIDDENNODES], double biasOutput[OUTPUTNODES]) {

    srand(time(NULL));

    for (int i = 0; i < HIDDENNODES; i++) {
        for (int j = 0; j < INPUTNODES; j++) {
            weightInputHidden[i][j] = 2.0 * (rand() / (double)RAND_MAX) - 1.0;
        }

        biasHidden[i] = 0.0;
    }

    for (int l = 0; l < OUTPUTNODES; l++) {
        for (int k = 0; k < HIDDENNODES; k++) {
            weightHiddenOutput[l][k] = 2.0 * (rand() / (double)RAND_MAX) - 1.0;
        }
        biasOutput[l] = 0.0;
    }
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double product(double* matrix, double* vector, int rows, int cols) {
    double result = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result += matrix[i * cols + j] * vector[j];
        }
    }
    return result;
}

double forwardPass(double* input, double weightInputHidden[HIDDENNODES][INPUTNODES],
                   double weightHiddenOutput[OUTPUTNODES][HIDDENNODES],
                   double biasHidden[HIDDENNODES], double biasOutput[OUTPUTNODES]) {

    double hiddenInputs[HIDDENNODES];
    for (int i = 0; i < HIDDENNODES; i++) {
        hiddenInputs[i] = product(weightInputHidden[i], input, 1, INPUTNODES) + biasHidden[i];
    }

    double hiddenOutputs[HIDDENNODES];
    for (int i = 0; i < HIDDENNODES; i++) {
        hiddenOutputs[i] = sigmoid(hiddenInputs[i]);
    }

    double outputInputs[OUTPUTNODES];
    for (int i = 0; i < OUTPUTNODES; i++) {
        outputInputs[i] = product(weightHiddenOutput[i], hiddenOutputs, 1, HIDDENNODES) + biasOutput[i];
    }

    double predictedOutput = sigmoid(outputInputs[0]);
    return predictedOutput;
}

void backwardPass(double* input, double output, double weightInputHidden[HIDDENNODES][INPUTNODES],
                  double weightHiddenOutput[OUTPUTNODES][HIDDENNODES],
                  double biasHidden[HIDDENNODES], double biasOutput[OUTPUTNODES],
                  double learning_rate) {

    double hiddenOutputs[HIDDENNODES];
    double hiddenInputs[HIDDENNODES];
    double outputInputs[OUTPUTNODES];
    double predictedOutput;

    for (int i = 0; i < HIDDENNODES; i++) {
        hiddenInputs[i] = product(weightInputHidden[i], input, 1, INPUTNODES) + biasHidden[i];
        hiddenOutputs[i] = sigmoid(hiddenInputs[i]);
    }

    for (int i = 0; i < OUTPUTNODES; i++) {
        outputInputs[i] = product(weightHiddenOutput[i], hiddenOutputs, 1, HIDDENNODES) + biasOutput[i];
    }

    predictedOutput = sigmoid(outputInputs[0]);

    // Calculate the error
    double error = output - predictedOutput;

    // Calculate the gradients
    double outputGradients = error * (predictedOutput * (1 - predictedOutput));

    double hiddenGradients[HIDDENNODES];
    for (int i = 0; i < HIDDENNODES; i++) {
        hiddenGradients[i] = outputGradients * weightHiddenOutput[0][i] *
                             (hiddenOutputs[i] * (1 - hiddenOutputs[i]));
    }

    // Update weights and biases for output layer
    for (int i = 0; i < OUTPUTNODES; i++) {
        for (int j = 0; j < HIDDENNODES; j++) {
            weightHiddenOutput[i][j] += learning_rate * outputGradients * hiddenOutputs[j];
        }
        biasOutput[i] += learning_rate * outputGradients;
    }

    // Update weights and biases for hidden layer
    for (int i = 0; i < HIDDENNODES; i++) {
        for (int j = 0; j < INPUTNODES; j++) {
            weightInputHidden[i][j] += learning_rate * hiddenGradients[i] * input[j];
        }
        biasHidden[i] += learning_rate * hiddenGradients[i];
    }
}

int main() {

    double weightInputHidden[HIDDENNODES][INPUTNODES];
    double weightHiddenOutput[OUTPUTNODES][HIDDENNODES];
    double biasHidden[HIDDENNODES];
    double biasOutput[OUTPUTNODES];

    initialize_weights_biases(weightInputHidden, weightHiddenOutput, biasHidden, biasOutput);

    printf("Weight Input Hidden:\n");
    for (int i = 0; i < HIDDENNODES; i++) {
        for (int j = 0; j < INPUTNODES; j++) {
            printf("%.2f ", weightInputHidden[i][j]);
        }
        printf("\n");
    }

    printf("Weight Hidden Output:\n");
    for (int l = 0; l < OUTPUTNODES; l++) {
        for (int k = 0; k < HIDDENNODES; k++) {
            printf("%.2f ", weightHiddenOutput[l][k]);
        }
        printf("\n");
    }

    printf("Bias Hidden: %.2f %.2f\n", biasHidden[0], biasHidden[1]);
    printf("Bias Output: %.2f\n", biasOutput[0]);

    // Training loop
    for (int epoch = 0; epoch < epochs_count; epoch++) {
        for (int j = 0; j < 4; j++) {
            double* input = xor_inputs[j];
            double output = xor_outputs[j];

            backwardPass(input, output, weightInputHidden, weightHiddenOutput, biasHidden, biasOutput, learning_rate);
        }
    }

    printf("Trained Weight Input Hidden:\n");
    for (int i = 0; i < HIDDENNODES; i++) {
        for (int j = 0; j < INPUTNODES; j++) {
            printf("%.2f ", weightInputHidden[i][j]);
        }
        printf("\n");
    }

    printf("Trained Weight Hidden Output:\n");
    for (int l = 0; l < OUTPUTNODES; l++) {
        for (int k = 0; k < HIDDENNODES; k++) {
            printf("%.2f ", weightHiddenOutput[l][k]);
        }
        printf("\n");
    }

    printf("Trained Bias Hidden: %.2f %.2f\n", biasHidden[0], biasHidden[1]);
    printf("Trained Bias Output: %.2f\n", biasOutput[0]);

    printf("Input 0 0, Predicted Output: %.2f\n", forwardPass(xor_inputs[0], weightInputHidden,
                                                               weightHiddenOutput, biasHidden, biasOutput));
    printf("Input 0 1, Predicted Output: %.2f\n", forwardPass(xor_inputs[1], weightInputHidden,
                                                               weightHiddenOutput, biasHidden, biasOutput));
    printf("Input 1 0, Predicted Output: %.2f\n", forwardPass(xor_inputs[2], weightInputHidden,
                                                               weightHiddenOutput, biasHidden, biasOutput));
    printf("Input 1 1, Predicted Output: %.2f\n", forwardPass(xor_inputs[3], weightInputHidden,
                                                               weightHiddenOutput, biasHidden, biasOutput));

    return 0;
}
