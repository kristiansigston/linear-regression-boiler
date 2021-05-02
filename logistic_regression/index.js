require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('../load-csv')
const LogisticRegression = require('./logistic_regression')
const plot = require('node-remote-plot')

const { features, labels, testFeatures, testLabels } = loadCSV(
  './data/cars.csv',
  {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['passedemissions'],
    converters: {
      passedemissions: (value) => {
        return value === 'TRUE' ? 1 : 0
      },
    },
  }
)

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
  decisionBoundary: 0.1,
})

regression.train()

regression
  .predict([
    [130, 307, 1.75],
    [88, 97, 1.065],
  ])
  .print()

console.log('res', regression.test(testFeatures, testLabels))

plot({
  x: regression.costHistory,
})
