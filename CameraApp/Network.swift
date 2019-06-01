//
//  Network.swift
//  MachineLearningApp
//
//  Created by Filip Osowski on 2/28/19.
//  Copyright © 2019 Filip Osowski. All rights reserved.
//

import Foundation
import Surge
import GameplayKit

infix operator +

func +(left: [[[Float]]], right: [[[Float]]]) -> [[[Float]]] {
    return zip(left, right).map({(arg_layer) -> [[Float]] in
        let (a, b) = arg_layer
        return zip(a, b).map({(arg) -> [Float] in
            let (x, y) = arg
            return Surge.add(x, y: y)
        })
    })
}

func +(left: [[Float]], right: [[Float]]) -> [[Float]] {
    return zip(left, right).map({(arg) -> [Float] in
        let (left, right) = arg
        return Surge.add(left, y: right)
    })
}

extension String {
    subscript (i: Int) -> Character {
        return self[index(startIndex, offsetBy: i)]
    }
    subscript (bounds: CountableRange<Int>) -> Substring {
        let start = index(startIndex, offsetBy: bounds.lowerBound)
        let end = index(startIndex, offsetBy: bounds.upperBound)
        return self[start ..< end]
    }
    subscript (bounds: CountableClosedRange<Int>) -> Substring {
        let start = index(startIndex, offsetBy: bounds.lowerBound)
        let end = index(startIndex, offsetBy: bounds.upperBound)
        return self[start ... end]
    }
    subscript (bounds: CountablePartialRangeFrom<Int>) -> Substring {
        let start = index(startIndex, offsetBy: bounds.lowerBound)
        let end = index(endIndex, offsetBy: -1)
        return self[start ... end]
    }
    subscript (bounds: PartialRangeThrough<Int>) -> Substring {
        let end = index(startIndex, offsetBy: bounds.upperBound)
        return self[startIndex ... end]
    }
    subscript (bounds: PartialRangeUpTo<Int>) -> Substring {
        let end = index(startIndex, offsetBy: bounds.upperBound)
        return self[startIndex ..< end]
    }
}
extension Substring {
    subscript (i: Int) -> Character {
        return self[index(startIndex, offsetBy: i)]
    }
    subscript (bounds: CountableRange<Int>) -> Substring {
        let start = index(startIndex, offsetBy: bounds.lowerBound)
        let end = index(startIndex, offsetBy: bounds.upperBound)
        return self[start ..< end]
    }
    subscript (bounds: CountableClosedRange<Int>) -> Substring {
        let start = index(startIndex, offsetBy: bounds.lowerBound)
        let end = index(startIndex, offsetBy: bounds.upperBound)
        return self[start ... end]
    }
    subscript (bounds: CountablePartialRangeFrom<Int>) -> Substring {
        let start = index(startIndex, offsetBy: bounds.lowerBound)
        let end = index(endIndex, offsetBy: -1)
        return self[start ... end]
    }
    subscript (bounds: PartialRangeThrough<Int>) -> Substring {
        let end = index(startIndex, offsetBy: bounds.upperBound)
        return self[startIndex ... end]
    }
    subscript (bounds: PartialRangeUpTo<Int>) -> Substring {
        let end = index(startIndex, offsetBy: bounds.upperBound)
        return self[startIndex ..< end]
    }
}


class Network {
    var manager: MNISTManager!
    var biases: [[Float]] = []
    var weights: [[[Float]]] = []
    var layers: [Int]
    let directory: String
    var data_last_loaded_with_batch_size = 0
    var output_label: UILabel!
    
    init(layers: [Int]) {
        self.layers = layers

        let random = GKRandomSource()
        let gaussian_distribution = GKGaussianDistribution(randomSource: random, mean: 0, deviation: 1)
        
        for layer in Array(layers[1...]) {
            var b: [Float] = []
            for _ in 0..<layer {
//                b.append(Float.random(in: -1...1))
                b.append(gaussian_distribution.nextUniform())
            }
            biases.append(b)
        }
        
        for (layer_index, layer_size) in Array(layers[1...]).enumerated() {
            var w_l: [[Float]] = []
            for _ in 0..<layer_size { // _ = neurons
                var w_j: [Float] = []
                for _ in 0..<layers[layer_index] { // _ = activations
//                    w_j.append(Float.random(in: -1...1))
                    w_j.append(gaussian_distribution.nextUniform())
                }
                w_l.append(w_j)
            }
            weights.append(w_l)
        }
        
        directory = String(Bundle.main.url(forResource: "MNIST", withExtension: nil)!.absoluteString[8...])
        
        print("Initialized network")
    }
    
    func setOutputLabel(output_label: UILabel) {
        self.output_label = output_label
    }
    
    func feedforward(activations: [Float]) -> [Float] {
        var a = activations
        
        for i in 0..<layers.count - 1 {
            let layer_weights = weights[i]
            let layer_biases = biases[i]
            a = Surge.add(layer_weights.map({Surge.dot($0, y: a)}), y: layer_biases).map({sigmoid($0)})
        }
        
        return a
    }
    
    func load_data(batchSize: Int) {
        if data_last_loaded_with_batch_size != batchSize {
            do {

                DispatchQueue.main.async {
                    self.output_label.text = "Loading data..."
                }
                print("Loading data...")
                manager = try MNISTManager(directory: directory, pixelRange: (0, 1), batchSize: batchSize)
                data_last_loaded_with_batch_size = batchSize
                DispatchQueue.main.async {
                    self.output_label.text = "Done loading data."
                }
                print("Done loading data.")
            } catch {
                print("Caught error")
                print(error)
            }
        }
    }
    
    func SGD_small(batch_number: Int, batch_size: Int, learning_rate: Float) {
        load_data(batchSize: batch_size)

        DispatchQueue.main.async {
            self.output_label.text = "Starting training..."
        }
        print("Starting training...")

        var batch_count = 0
        for (inputs, labels) in zip(manager.trainImages, manager.trainLabels).shuffled() {
            var nabla_w: [[[Float]]] = weights.map({$0.map({$0.map({(x) -> Float in 0})})})
            var nabla_b: [[Float]] = biases.map({$0.map({(x) -> Float in 0})})

            for (input, label) in zip(inputs, labels) {
                var delta_nabla_w: [[[Float]]]
                var delta_nabla_b: [[Float]]
                (delta_nabla_w, delta_nabla_b) = backprop(input, label)

                nabla_b = nabla_b + delta_nabla_b
                nabla_w = nabla_w + delta_nabla_w
            }


            weights = weights + nabla_w.map({$0.map({$0.map({$0 * -learning_rate/Float(batch_size)})})})
            biases = biases + nabla_b.map({$0.map({$0 * -learning_rate/Float(batch_size)})})

            batch_count += 1

            if batch_count == batch_number {
                print("Finished training on", batch_number, "batches")
                evaluate_model()
                break
            }
        }
    }
    
    func SGD(epochs: Int, batch_size: Int, learning_rate: Float) {
        load_data(batchSize: batch_size)

        print("Starting training...")
        for epoch_index in 0..<epochs {
            print("Epoch number", epoch_index)

            var batch_count = 0
            let evaluate_at = 100
            var examples_evaluated = 0
            for (inputs, labels) in zip(manager.trainImages, manager.trainLabels).shuffled() {
                var nabla_w: [[[Float]]] = weights.map({$0.map({$0.map({(x) -> Float in 0})})})
                var nabla_b: [[Float]] = biases.map({$0.map({(x) -> Float in 0})})

                for (input, label) in zip(inputs, labels) {
                    var delta_nabla_w: [[[Float]]]
                    var delta_nabla_b: [[Float]]
                    (delta_nabla_w, delta_nabla_b) = backprop(input, label)

                    nabla_b = nabla_b + delta_nabla_b
                    nabla_w = nabla_w + delta_nabla_w
                    examples_evaluated += 1
                }


                weights = weights + nabla_w.map({$0.map({$0.map({$0 * -learning_rate/Float(batch_size)})})})
                biases = biases + nabla_b.map({$0.map({$0 * -learning_rate/Float(batch_size)})})

                batch_count += 1

                if batch_count == evaluate_at {
                    evaluate_model()
                    print("Examples evaluated", examples_evaluated)
                    batch_count = 0
                }
            }
        }
    }
    
    func max_index(_ arr: [Float]) -> Int {
        var max = arr[0]
        var max_i = 0
        var i = 0
        
        for x in arr {
            if x > max {
                max = x
                max_i = i
            }
            i += 1
        }
        
        return max_i
    }
    
    func evaluate_model() {
        var correct = 0
        var total = 0
        
        for (inputs, labels) in zip(manager.validationImages, manager.validationLabels) {
            for (input, label) in zip(inputs, labels) {
                let output = self.feedforward(activations: input)
                let x = max_index(output)
                let y = max_index(label)
                
                if x == y {
                    correct += 1
                }
                total += 1
            }
        }
        DispatchQueue.main.async {
            self.output_label.text = "Accuracy: " + String(Float(correct)/Float(total))
        }
        print("Percentage Correct-", correct, "/", total)
    }
    
    func backprop(_ input: [Float], _ label: [Float]) -> ([[[Float]]], [[Float]]) {
        var z_s: [[Float]] = []
        var a_s: [[Float]] = []

        var a = input
        a_s.append(a)
        
        // Storing unweighted inputs (z's) and activations (a's)
        for i in 0..<layers.count - 1 {
            let z = Surge.add(weights[i].map({Surge.dot($0, y: a)}), y: biases[i])
            a = z.map({sigmoid($0)})
            z_s.append(z)
            a_s.append(a)
        }

        var deltas: [[Float]] = biases.map({$0.map({(x) -> Float in 0})})
        var nabla_w: [[[Float]]] = weights.map({$0.map({$0.map({(x) -> Float in 0})})})
        var nabla_b: [[Float]] = biases.map({$0.map({(x) -> Float in 0})})
        
        // Calculating delta for last layer
        let last_delta = zip(Surge.add(a, y: label.map({$0 * -1})), z_s[z_s.count - 1].map({sigmoid_prime($0)})).map({$0 * $1})
        deltas[deltas.count - 1] = last_delta
        
        // Calculating all deltas for all other layers
        for i in stride(from: deltas.count - 2, to: -1, by: -1) {
            let delta = zip(transpose(weights[i + 1]).map({Surge.dot($0, y: deltas[i + 1])}), z_s[i].map({sigmoid_prime($0)})).map({$0 * $1})
            deltas[i] = delta
        }
        
        // Calculating nabla_b and nabla_w for all layers
        for i in 0..<deltas.count {
            nabla_b[i] = deltas[i]
            nabla_w[i] = deltas[i].map({(e) -> [Float] in
                return a_s[i].map({$0 * e})
            })
        }

        return (nabla_w, nabla_b)
    }

    func sigmoid(_ z: Float) -> Float {
        return (1/(1 + pow(Float(M_E), -z)))
    }
    
    func sigmoid_prime(_ z: Float) -> Float {
        return sigmoid(z) * (1 - sigmoid(z))
    }
    
    public func transpose<T>(_ input: [[T]]) -> [[T]] {
        if input.isEmpty { return [[T]]() }
        let count = input[0].count
        var out = [[T]](repeating: [T](), count: count)
        for outer in input {
            for (index, inner) in outer.enumerated() {
                out[index].append(inner)
            }
        }
        
        return out
    }
}
