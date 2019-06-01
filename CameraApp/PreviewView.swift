//
//  PreviewView.swift
//  CameraApp
//
//  Created by Cat Blue on 2/15/19.
//  Copyright © 2019 Filip Osowski. All rights reserved.
//

import Foundation
import UIKit
import AVFoundation

class PreviewView: UIView {
    override class var layerClass: AnyClass {
        return AVCaptureVideoPreviewLayer.self
    }
    
    /// Convenience wrapper to get layer as its statically known type.
    var videoPreviewLayer: AVCaptureVideoPreviewLayer {
        return layer as! AVCaptureVideoPreviewLayer
    }
}
