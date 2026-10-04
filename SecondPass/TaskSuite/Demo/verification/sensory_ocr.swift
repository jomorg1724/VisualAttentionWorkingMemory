import Foundation
import PDFKit
import Vision
import AppKit
let args = CommandLine.arguments
for path in args.dropFirst() {
 guard let doc = PDFDocument(url: URL(fileURLWithPath:path)) else {continue}
 var text = ""
 for i in 0..<doc.pageCount {
  autoreleasepool {
   guard let page = doc.page(at:i) else {return}
   let box = page.bounds(for:.mediaBox)
   let scale: CGFloat = 2
   guard let ctx = CGContext(data:nil,width:Int(box.width*scale),height:Int(box.height*scale),bitsPerComponent:8,bytesPerRow:0,space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue) else {return}
   ctx.setFillColor(CGColor(gray:1,alpha:1));ctx.fill(CGRect(x:0,y:0,width:box.width*scale,height:box.height*scale))
   ctx.scaleBy(x:scale,y:scale); page.draw(with:.mediaBox,to:ctx)
   guard let image = ctx.makeImage() else {return}
   let request = VNRecognizeTextRequest();request.recognitionLevel = .accurate;request.usesLanguageCorrection = false; request.usesCPUOnly = true
   do {try VNImageRequestHandler(cgImage:image).perform([request])} catch {print(error);return}
   text += "\n=== PDF PAGE \(i+1) ===\n" + (request.results ?? []).compactMap{$0.topCandidates(1).first?.string}.joined(separator:"\n") + "\n"
  }
 }
 try text.write(toFile:path.replacingOccurrences(of:".pdf",with:".txt"),atomically:true,encoding:.utf8)
 print("OCR \(doc.pageCount) pages: \(path)")
}
