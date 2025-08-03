import * as tf from '@tensorflow/tfjs';

// Define the machine learning model
class MachineLearningModel {
  private model: tf.Sequential;

  constructor() {
    this.model = tf.sequential();
    this.model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
    this.model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
  }

  async predict(input: number): Promise<number> {
    const output = this.model.predict(tf.tensor2d([input], [1, 1]));
    return output.dataSync()[0];
  }
}

// Define the notifier class
class Notifier {
  private model: MachineLearningModel;

  constructor(model: MachineLearningModel) {
    this.model = model;
  }

  notify(input: number): void {
    console.log(`Received input: ${input}`);
    this.model.predict(input).then((output) => {
      console.log(`Predicted output: ${output}`);
      if (output > 0.5) {
        console.log('Notification triggered!');
      }
    });
  }
}

// Create an instance of the machine learning model
const mlModel = new MachineLearningModel();

// Create an instance of the notifier
const notifier = new Notifier(mlModel);

// Mock some input data
const inputs = [0.1, 0.3, 0.5, 0.7, 0.9];

// Trigger the notifier for each input
inputs.forEach((input) => {
  notifier.notify(input);
});