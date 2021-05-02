const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

const defaultOptions = {
  learningRate: 0.1,
  iterations: 1000,
  batchSize: 1,
  decisionBoundary: 0.5,
}

class LogisticRegression {
  constructor(features, labels, options) {
    // add the 1s column and make a tensor
    this.features = this.processFeatures(features)
    this.labels = tf.tensor(labels)
    this.costHistory = []

    this.options = Object.assign(defaultOptions, options)

    this.weights = tf.zeros([this.features.shape[1], 1])
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights).sigmoid()
    const differences = currentGuesses.sub(labels)

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0])

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate))
  }

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

      this.recordCost()
      this.updateLearningRate()
    }
  }

  predict(observations) {
    const results = this.processFeatures(observations)
      .matMul(this.weights)
      .sigmoid()
      .greater(this.options.decisionBoundary)
      .cast('float32')
    return results
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures)
    testLabels = tf.tensor(testLabels)

    const incorrect = predictions.sub(testLabels).abs().sum().get()

    return (1 - incorrect / predictions.shape[0]) * 100
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

  recordCost() {
    const guesses = this.features.matMul(this.weights).sigmoid()
    const term1 = this.labels.transpose().matMul(guesses.log())
    const term2 = this.labels
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(guesses.mul(-1).add(1).log())

    const cost = term1.add(term2).div(this.features.shape[0]).mul(-1).get(0, 0)

    this.costHistory.push(cost)
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) return
    const lastValue = this.costHistory[this.costHistory.length - 1]
    const secondValue = this.costHistory[this.costHistory.length - 2]

    if (lastValue < secondValue) {
      this.options.learningRate *= 1.05
    } else this.options.learningRate /= 2
  }
}

module.exports = LogisticRegression

// new linearRegression(features, labels, {
//   iterations: 99,
//   learningRate: 0.01
// })
