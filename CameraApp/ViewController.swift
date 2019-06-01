//
//  ViewController.swift
//  CameraApp
//
//  Created by Cat Blue on 2/15/19.
//  Copyright © 2019 Filip Osowski. All rights reserved.
//

import UIKit
import AVFoundation
import CoreMotion

class ViewController: UIViewController {
    
    let captureSession = AVCaptureSession()
    let previewView = PreviewView()
    let motionManager = CMMotionManager()
    let answerLabel = UILabel()
    let cameraButton = UIButton(frame: CGRect(x: 700, y: 100, width: 200, height: 100))
    let networkButton = UIButton(frame: CGRect(x: 700, y: 500, width: 200, height: 100))
//    let rectangle = UIView(frame: CGRect(x: 0, y: 0, width: 336, height: 336))
    let rectangle = UIView(frame: CGRect(x: 0, y: 0, width: 336, height: 336))
    let updateRate = 0.1
    let photoOutput = AVCapturePhotoOutput()
    let network = Network(layers: [784, 30, 10])
    
    var count = 0
    var timer: Timer!
    var lastAvgMagnitude = 0.0
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        previewView.frame = self.view.frame
        previewView.backgroundColor = UIColor.darkGray
        self.view.addSubview(previewView)
        
        cameraButton.setTitle("PREDICT", for: .normal)
        cameraButton.setTitleColor(.black, for: .normal)
        cameraButton.backgroundColor = UIColor.lightGray
        cameraButton.addTarget(self, action: #selector(takePhoto), for: .touchUpInside)
        self.view.addSubview(cameraButton)
        
        networkButton.setTitle("TRAIN", for: .normal)
        networkButton.setTitleColor(.black, for: .normal)
        networkButton.backgroundColor = UIColor.lightGray
        networkButton.addTarget(self, action: #selector(trainNetwork), for: .touchUpInside)
        self.view.addSubview(networkButton)
        
        let x = self.view.frame.width * 336/3264
        rectangle.frame = CGRect(x: 0, y: 0, width: x, height: x)
        rectangle.backgroundColor = .red
        rectangle.alpha = 0.25
        self.view.addSubview(rectangle)
        
        answerLabel.text = "#"
        answerLabel.alpha = 1
        answerLabel.textColor = UIColor.red
        answerLabel.frame = CGRect(x: self.view.frame.size.height/2, y: self.view.frame.size.width/2, width: 800, height: 300)
        answerLabel.textAlignment = .center
        answerLabel.center.x = self.view.center.x
        answerLabel.center.y = self.view.center.y
        answerLabel.font = UIFont(name: "Helvetica-Bold", size: 70.0)
        self.view.addSubview(answerLabel)
        network.setOutputLabel(output_label: answerLabel)

        captureSession.beginConfiguration()
        captureSession.sessionPreset = .photo
        let captureDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .unspecified)
        let captureDeviceInput = try? AVCaptureDeviceInput(device: captureDevice!)
        captureSession.addInput(captureDeviceInput!)
        captureSession.addOutput(photoOutput)
        captureSession.commitConfiguration()
        
        previewView.videoPreviewLayer.session = self.captureSession
        previewView.backgroundColor = UIColor.clear
        previewView.videoPreviewLayer.connection?.videoOrientation = .landscapeRight
        
        captureSession.startRunning()
    }
    
    @objc func trainNetwork() {
//        network.SGD(epochs: 1, batch_size: 10, learning_rate: 3)
        self.answerLabel.text = "Training"
        DispatchQueue.global().async {
            self.network.SGD_small(batch_number: 100, batch_size: 10, learning_rate: 3)
        }
    }
    
    @objc func takePhoto() {
        if let photoOutputConnection = self.photoOutput.connection(with: .video) {
            photoOutputConnection.videoOrientation = .landscapeRight
        }
        
        var photoSettings = AVCapturePhotoSettings()
        
        // Capture HEIF photos when supported. Enable auto-flash and high-resolution photos.
        if self.photoOutput.availablePhotoCodecTypes.contains(.hevc) {
            photoSettings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.hevc])
        }
        
        self.photoOutput.capturePhoto(with: photoSettings, delegate: self)
    }
}

extension ViewController: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let imageData = photo.fileDataRepresentation() {
            if let image = UIImage(data: imageData){
                var values_for_scaled_image: [Float] = [0]
                
                let mapped_values: [[Float]] = [[1, 68], [0.85, 20], [0.0, 705]]
                
                for value in mapped_values {
                    values_for_scaled_image += Array(repeating: value[0], count: Int(value[1]))
                }

//                let downsized_image = image.resized(toWidth: 272)!
                let true_downsized_image = resize(image.cgImage!)
                let downsized_image = UIImage(cgImage: true_downsized_image!)
                let cropped_true_downsized_image = true_downsized_image!.cropping(to: CGRect(x: 0, y: 0, width: 28, height: 28))!
                let number_image = UIImage(cgImage: cropped_true_downsized_image)
                
                
//                let provider = cropped_true_downsized_image.dataProvider
//                let providerData = provider!.data
//                let dataPtr = CFDataGetBytePtr(providerData)
//
//                let bytesPerRow = cropped_true_downsized_image.bytesPerRow
//                var data: [GLubyte] = Array(UnsafeBufferPointer(start: dataPtr, count: cropped_true_downsized_image.height * bytesPerRow))
//
//                print("DATA IS")
//                print(data)
                
                var pixels: [Float] = []
                for y in 0..<28 {
                    for x in 0..<28 {
//                        let color = number_image.getPixelColor(x: x, y: y)!.cgColor.components!
                        let color = number_image[x, y]?.cgColor.components!
//                        let scale: Float = Float((color[0] + color[1] + color[2]) / 3)
//                        print(x, y)
//                        print(color)
                        let grayscale = 1 - (Float(color![0]) + Float(color![1]) + Float(color![2]))/3
                        pixels.append(grayscale)
//                        pixels.append(scale)
                    }
                }

                let sorted_pixels = pixels.enumerated().sorted(by: {(x, y) -> Bool in
                    return x.element > y.element
                }).map({$0.offset})

                for (index, element_index) in sorted_pixels.enumerated() {
                    pixels[element_index] = values_for_scaled_image[index]
                }
                
                print("AFTER")
                print(pixels)

//                print("SCALED PIXELS ARE", scaled_pixels)
//                print("OUTPUT IS", output)
                let answer = network.max_index(network.feedforward(activations: pixels))
                print("ANSWER IS", answer)
                answerLabel.text = String(answer)
                
//                UIImageWriteToSavedPhotosAlbum(downsized_image, nil, nil, nil)
            }
        }
    }

    func resize(_ image: CGImage) -> CGImage? {
        var ratio: Float = 1/12
        let imageWidth = Float(image.width)
        let imageHeight = Float(image.height)
        
        if ratio > 1 {
            ratio = 1
        }
        
        let width = imageWidth * ratio
        let height = imageHeight * ratio

        guard let colorSpace = image.colorSpace else { return nil }
        guard let context = CGContext(data: nil, width: Int(width), height: Int(height), bitsPerComponent: image.bitsPerComponent, bytesPerRow: image.bytesPerRow, space: colorSpace, bitmapInfo: image.alphaInfo.rawValue) else { return nil }
        
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: Int(width), height: Int(height)))
        
        return context.makeImage()
        
    }
}

extension UIImage {
    
    subscript (x: Int, y: Int) -> UIColor? {
        
        if x < 0 || x > Int(size.width) || y < 0 || y > Int(size.height) {
            return nil
        }
        
        let provider = self.cgImage!.dataProvider
        let providerData = provider!.data
        let data = CFDataGetBytePtr(providerData)
        
//        let numberOfComponents = 4
        let bytesPerRow = self.cgImage!.bytesPerRow
        let pixelData = bytesPerRow * y + 4 * x
        
        let r = CGFloat(data![pixelData]) / 255.0
        let g = CGFloat(data![pixelData + 1]) / 255.0
        let b = CGFloat(data![pixelData + 2]) / 255.0
        let a = CGFloat(data![pixelData + 3]) / 255.0
        
        return UIColor(red: r, green: g, blue: b, alpha: a)
    }
}
