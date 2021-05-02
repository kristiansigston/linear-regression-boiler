const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

const defaultOptions = { learningRate: 0.1, iterations: 1000, batchSize: 1 }

class linearRegression {
  constructor(features, labels, options) {
    // add the 1s column and make a tensor
    this.features = this.processFeatures(features)
    this.labels = tf.tensor(labels)
    this.mseHistory = []

    this.options = Object.assign(defaultOptions, options)

    console.log('this.features.shape', this.features.shape)
    this.weights = tf.zeros([this.features.shape[1], 1])
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights)
    const differences = currentGuesses.sub(labels)

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0])

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate))
  }

  // gradientDescent() {
  //   let currentGuessForMPG = this.features.map(
  //     (row) => this.m * row[0] + this.b
  //   )
  //   const samples = this.features.length

  //   const bSlope =
  //     (currentGuessForMPG.reduce((a, b, index) => {
  //       return a + b - this.labels[index][0]
  //     }, 0) *
  //       2) /
  //     samples

  //   const mSlope =
  //     (currentGuessForMPG.reduce((a, b, index) => {
  //       return -1 * this.features[index][0] * (this.labels[index][0] - b) + a
  //     }, 0) *
  //       2) /
  //     samples

  //   this.m = this.m - this.options.learningRate * mSlope
  //   this.b = this.b - this.options.learningRate * bSlope
  // }

  train() {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    )
    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const { batchSize } = this.options
        const startIndex = j * batchSize
        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1]
        )
        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1])
        this.gradientDescent(featureSlice, labelSlice)
      }

      this.recordMSE()
      this.updateLearningRate()
    }
  }

  predict(observations) {
    return this.processFeatures(observations).matMul(this.weights)
  }

  test(testFeatures, testLabels) {
    testFeatures = this.processFeatures(testFeatures)
    testLabels = tf.tensor(testLabels)

    const predictions = testFeatures.matMul(this.weights)

    const res = testLabels.sub(predictions).pow(2).sum().get()

    const tot = testLabels.sub(testLabels.mean()).pow(2).sum().get()

    return 1 - res / tot
  }

  processFeatures(features) {
    features = tf.tensor(features)

    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.sqrt())
    } else {
      features = this.standarize(features)
    }
    features = tf.ones([features.shape[0], 1]).concat(features, 1)
    return features
  }

  standarize(features) {
    const { mean, variance } = tf.moments(features, 0)

    this.mean = mean
    this.variance = variance

    return features.sub(mean).div(variance.sqrt())
  }

  recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .get()
    this.mseHistory.push(mse)
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) return
    const lastValue = this.mseHistory[this.mseHistory.length - 1]
    const secondValue = this.mseHistory[this.mseHistory.length - 2]

    if (lastValue < secondValue) {
      this.options.learningRate *= 1.05
    } else this.options.learningRate /= 2
  }
}

module.exports = linearRegression

// new linearRegression(features, labels, {
//   iterations: 99,
//   learningRate: 0.01
// })
