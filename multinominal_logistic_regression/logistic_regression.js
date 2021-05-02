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

    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]])
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights).softmax()
    const differences = currentGuesses.sub(labels)

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0])

    return this.weights.sub(slopes.mul(this.options.learningRate))
  }

  train() {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    )
    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const { batchSize } = this.options
        const startIndex = j * batchSize

        this.weights = tf.tidy(() => {
          const featureSlice = this.features.slice(
            [startIndex, 0],
            [batchSize, -1]
          )
          const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1])
          return this.gradientDescent(featureSlice, labelSlice)
        })
      }
      this.recordCost()
      this.updateLearningRate()
    }
  }

  predict(observations) {
    const results = this.processFeatures(observations)
      .matMul(this.weights)
      .softmax()
      .argMax(1)
    return results
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures)
    testLabels = tf.tensor(testLabels).argMax(1)

    const incorrect = predictions.notEqual(testLabels).sum().get()

    return (1 - incorrect / predictions.shape[0]) * 100
  }

  processFeatures(features) {
    features = tf.tensor(features)
    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.sqrt())
    } else {
      features = this.standardize(features)
    }
    features = tf.ones([features.shape[0], 1]).concat(features, 1)
    return features
  }

  standardize(features) {
    const { mean, variance } = tf.moments(features, 0)

    // deal with zero variance
    const filler = variance.cast('bool').logicalNot().cast('float32')

    this.mean = mean
    this.variance = variance.add(filler)
    return features.sub(mean).div(this.variance.pow(0.5))
  }

  recordCost() {
    const cost = tf.tidy(() => {
      const guesses = this.features.matMul(this.weights).softmax()
      const term1 = this.labels.transpose().matMul(guesses.add(1e-7).log())
      const term2 = this.labels
        .mul(-1)
        .add(1)
        .transpose()
        .matMul(guesses.mul(-1).add(1).add(1e-7).log()) // add 0.0000001 so we never take a log of 0

      return term1.add(term2).div(this.features.shape[0]).mul(-1).get(0, 0)
    })

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
