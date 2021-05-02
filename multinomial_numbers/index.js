require('@tensorflow/tfjs-node')
const LogisticRegression = require('../multinominal_logistic_regression/logistic_regression')
const tf = require('@tensorflow/tfjs')
const plot = require('node-remote-plot')
const _ = require('lodash')
const mnist = require('mnist-data')

function loadData() {
  const mnistData = mnist.training(0, 60000)

  const features = mnistData.images.values.map((image) => _.flatMap(image))

  const encodedLabels = mnistData.labels.values.map((label) => {
    const row = new Array(10).fill(0)
    row[label] = 1
    return row
  })
  return { features, labels: encodedLabels }
}

const { features, labels } = loadData()

const regression = new LogisticRegression(features, labels, {
  learningRate: 1,
  iterations: 40,
  batchSize: 250,
})

regression.train()

function loadTestData() {
  // inside a function to let go of the the testmindata ref
  const testMnistData = mnist.testing(0, 10000)

  const testFeatures = testMnistData.images.values.map((image) =>
    _.flatMap(image)
  )
  const testEncodedLabels = testMnistData.labels.values.map((label) => {
    const row = new Array(10).fill(0)
    row[label] = 1
    return row
  })
  return { testFeatures, testLabels: testEncodedLabels }
}

const { testFeatures, testLabels } = loadTestData()

const accuracy = regression.test(testFeatures, testLabels)
console.log('accuracy', accuracy)
plot({
  x: regression.costHistory,
})
