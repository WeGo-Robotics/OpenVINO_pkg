#!/usr/bin/env python3

"""
 Copyright (C) 2018-2022 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1] / 'open_model_zoo/openvino_demos/common/python'))

from openvino.model_zoo.model_api.models import DetectionModel, DetectionWithLandmarks, RESIZE_TYPES, OutputTransform
from openvino.model_zoo.model_api.performance_metrics import PerformanceMetrics
from openvino.model_zoo.model_api.pipelines import get_user_config, AsyncPipeline
from openvino.model_zoo.model_api.adapters import create_core, OpenvinoAdapter
from visualizers import ColorPalette
from helpers import resolution

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from object_msgs.msg import ObjectArray, Object 

def log_adapter_configuration(adapter, model, config):
    try:
        from openvino.runtime import get_version
        openvino_absent = False
    except ImportError:
        openvino_absent = True
    rospy.loginfo('OpenVINO Runtime')
    rospy.loginfo('\tbuild: {}'.format(get_version()))
    
    rospy.logwarn('Reading model {}'.format('from buffer' if adapter.model_from_buffer else adapter.model_path))

    if config is not None:
        parameters = model.parameters()
        for name, value in config.items():
            if name in parameters:
                errors = parameters[name].validate(value)
                if errors:
                    rospy.logerr(f'Error with "{name}" parameter:')
                    for error in errors:
                        rospy.logerr(f"\t{error}")
                    model.raise_error('Incorrect user configuration')
                value = parameters[name].get_value(value)
                model.__setattr__(name, value)
            else:
                rospy.logwarn(f'The parameter "{name}" not found in {model.__model__} wrapper, will be omitted')

    rospy.loginfo('The model {} is loaded to {}'.format("from buffer" if adapter.model_from_buffer else adapter.model_path, adapter.device))
    
def log_latency_per_stage(*pipeline_metrics):
    stages = ('Preprocessing', 'Inference', 'Postprocessing', 'Rendering')
    for stage, latency in zip(stages, pipeline_metrics):
        rospy.loginfo('\t{}:\t{:.1f} ms'.format(stage, latency))

def metrics_log_total(metrics):
        total_latency, total_fps = metrics.get_total()

        print('\n')
        rospy.loginfo('Metrics report:')
        rospy.loginfo("\tLatency: {:.1f} ms".format(total_latency * 1e3) if total_latency is not None else "\tLatency: N/A")
        rospy.loginfo("\tFPS: {:.1f}".format(total_fps) if total_fps is not None else "\tFPS: N/A")

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', required=True,
                      help='Required. Path to an .xml file with a trained model '
                           'or address of model inference service if using ovms adapter.')
    available_model_wrappers = [name.lower() for name in DetectionModel.available_wrappers()]
    args.add_argument('-at', '--architecture_type', help='Required. Specify model\' architecture type.',
                      type=str, required=True, choices=available_model_wrappers)
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is '
                           'acceptable. The demo will look for a suitable plugin for device specified. '
                           'Default value is CPU.')

    common_model_args = parser.add_argument_group('Common model options')
    common_model_args.add_argument('--labels', help='Optional. Labels mapping file.', default=None, type=str)
    common_model_args.add_argument('-t', '--prob_threshold', default=0.5, type=float,
                                   help='Optional. Probability threshold for detections filtering.')
    common_model_args.add_argument('--resize_type', default=None, choices=RESIZE_TYPES.keys(),
                                   help='Optional. A resize type for model preprocess. By default used model predefined type.')
    common_model_args.add_argument('--input_size', default=(600, 600), type=int, nargs=2,
                                   help='Optional. The first image size used for CTPN model reshaping. '
                                        'Default: 600 600. Note that submitted images should have the same resolution, '
                                        'otherwise predictions might be incorrect.')
    common_model_args.add_argument('--anchors', default=None, type=float, nargs='+',
                                   help='Optional. A space separated list of anchors. '
                                        'By default used default anchors for model. Only for YOLOV4 architecture type.')
    common_model_args.add_argument('--masks', default=None, type=int, nargs='+',
                                   help='Optional. A space separated list of mask for anchors. '
                                        'By default used default masks for model. Only for YOLOV4 architecture type.')
    common_model_args.add_argument('--layout', type=str, default=None,
                                   help='Optional. Model inputs layouts. '
                                        'Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.')
    common_model_args.add_argument('--num_classes', default=None, type=int,
                                   help='Optional. Number of detected classes. Only for NanoDet, NanoDetPlus '
                                        'architecture types.')

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                            default=0, type=int)
    infer_args.add_argument('-nstreams', '--num_streams',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).',
                            default='', type=str)
    infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('--no_show', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    input_transform_args = parser.add_argument_group('Input transform options')
    input_transform_args.add_argument('--reverse_input_channels', default=False, action='store_true',
                                      help='Optional. Switch the input channels order from '
                                           'BGR to RGB.')
    input_transform_args.add_argument('--mean_values', default=None, type=float, nargs=3,
                                      help='Optional. Normalize input by subtracting the mean '
                                           'values per channel. Example: 255.0 255.0 255.0')
    input_transform_args.add_argument('--scale_values', default=None, type=float, nargs=3,
                                      help='Optional. Divide input by scale values per channel. '
                                           'Division is applied after mean values subtraction. '
                                           'Example: 255.0 255.0 255.0')

    return parser

def draw_detections(frame, detections, palette, labels, output_transform):
    frame = output_transform.resize(frame)
    detectObjects = ObjectArray()
    for detection in detections:
        class_id = int(detection.id)
        color = palette[class_id]
        det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
        xmin, ymin, xmax, ymax = detection.get_coords()
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                    (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
        detectObject = Object(class_name = det_label, xmin_ymin_xmax_ymax=[xmin, ymin, xmax, ymax])
        # detectObject.class_name = det_label
        # detectObject.xmin_ymin_xmax_ymax = [xmin, ymin, xmax, ymax]
        detectObjects.Objects.append(detectObject)

        if isinstance(detection, DetectionWithLandmarks):
            for landmark in detection.landmarks:
                landmark = output_transform.scale(landmark)
                cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 255), 2)
    
    return frame, detectObjects

class inference_node(): 
    def __init__(self, rospyargs):
        
        # Inference engine setup
        # args = build_argparser().parse_args()
        args = rospyargs
        model, self.detector_pipeline = self.inference_engine_setup(args)
        palette = ColorPalette(len(model.labels) if model.labels else 100)

        # ImgMsg Info
        self.frame_shape = (480, 640)
        self.frame_id = ''
        self.seq = 0

        output_transform = OutputTransform(self.frame_shape, args.output_resolution)

        # ROS arguments setup      
        self.bridge = CvBridge()
        rospy.Subscriber('/image', Image, self.imgCb, queue_size=5)
        # rospy.Subscriber('/image', Image, self.imgCb, queue_size=5)
        # rospy.Subscriber('/image', CompressedImage, self.compressedImgCb, queue_size=5)

        framepub = rospy.Publisher('/vino/detect/image_raw', Image, queue_size=5)
        # frameCompressedpub = rospy.Publisher('/vino/detect/image_raw/compressed', CompressedImage, queue_size=5)
        objectpub = rospy.Publisher('/vino/detect/objects', ObjectArray, queue_size=5)

        self.next_frame_id = 0
        self.next_frame_id_to_show = 0

        # Metrics setup
        metrics = PerformanceMetrics()
        render_metrics = PerformanceMetrics()

        while not rospy.is_shutdown():
            if self.detector_pipeline.callback_exceptions:
                raise self.detector_pipeline.callback_exceptions[0]
                
            # Process all completed requests
            results = self.detector_pipeline.get_result(self.next_frame_id_to_show)
            if results:
                
                # Detection result disassemble
                objects, frame_meta = results
                frame = frame_meta['frame']
                start_time = frame_meta['start_time']
                
                # Frame rendering & metrics update
                rendering_start_time = perf_counter()
                frame, detectObjects = draw_detections(frame, objects, palette, model.labels, output_transform)
                render_metrics.update(rendering_start_time)
                                
                # Publish detection image
                framepub.publish(self.frame_to_imgMsg(frame))
                # frameCompressedpub.publish(self.frame_to_compressedImgMsg(frame))
                
                # Publish detected objects
                objectpub.publish(detectObjects)

                metrics.update(start_time, frame)
                self.next_frame_id_to_show += 1

                # Show detection results 
                if not args.no_show:
                    cv2.imshow('Detection Results', frame)
                    cv2.waitKey(1)
                    
            else:
                # Wait for empty request
                self.detector_pipeline.await_any()
        
        cv2.destroyAllWindows()
        self.detector_pipeline.await_all()

        if self.detector_pipeline.callback_exceptions:
            raise self.detector_pipeline.callback_exceptions[0]

        # Process completed requests
        for next_frame_id_to_show in range(self.next_frame_id_to_show, self.next_frame_id):
            results = self.detector_pipeline.get_result(next_frame_id_to_show)
            objects, frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            rendering_start_time = perf_counter()
            frame, _ = draw_detections(frame, objects, palette, model.labels, output_transform)
            render_metrics.update(rendering_start_time)
            metrics.update(start_time, frame)
        
        metrics_log_total(metrics)
        log_latency_per_stage(self.detector_pipeline.preprocess_metrics.get_latency(),
                              self.detector_pipeline.inference_metrics.get_latency(),
                              self.detector_pipeline.postprocess_metrics.get_latency(),
                              render_metrics.get_latency())

    def imgCb(self, imgmsg):
        if self.detector_pipeline.is_ready() and self.next_frame_id < self.next_frame_id_to_show+2:
            start_time = perf_counter() 
            frame = self.bridge.imgmsg_to_cv2(imgmsg, 'bgr8')
            self.detector_pipeline.submit_data(frame, self.next_frame_id, {'frame': frame, 'start_time': start_time})
            self.next_frame_id += 1

    def compressedImgCb(self, imgmsg):
        if self.detector_pipeline.is_ready() and self.next_frame_id < self.next_frame_id_to_show+2:
            start_time = perf_counter()         
            frame = self.bridge.compressed_imgmsg_to_cv2(imgmsg)
            self.detector_pipeline.submit_data(frame, self.next_frame_id, {'frame': frame, 'start_time': start_time})
            self.next_frame_id += 1

    def frame_to_imgMsg(self, frame):
        imgMsg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        imgMsg.header.stamp = rospy.Time.now()
        imgMsg.header.frame_id = self.frame_id
        imgMsg.header.seq = self.seq
        self.seq+=1
        return imgMsg

    def frame_to_compressedImgMsg(self, frame):
        imgMsg = self.bridge.cv2_to_compressed_imgmsg(frame)
        imgMsg.header.stamp = rospy.Time.now()
        imgMsg.header.frame_id = self.frame_id
        imgMsg.header.seq = self.seq
        self.seq+=1
        return imgMsg     

    def inference_engine_setup(self, args):
        if args.architecture_type != 'yolov4' and args.anchors:
            rospy.logwarn('The "--anchors" option works only for "-at==yolov4". Option will be omitted')
        if args.architecture_type != 'yolov4' and args.masks:
            rospy.logwarn('The "--masks" option works only for "-at==yolov4". Option will be omitted')
        if args.architecture_type not in ['nanodet', 'nanodet-plus'] and args.num_classes:
            rospy.logwarn('The "--num_classes" option works only for "-at==nanodet" and "-at==nanodet-plus". Option will be omitted')

        plugin_config = get_user_config(args.device, args.num_streams, args.num_threads)
        model_adapter = OpenvinoAdapter(create_core(), args.model, device=args.device, plugin_config=plugin_config,
                                        max_num_requests=args.num_infer_requests, model_parameters = {'input_layouts': args.layout})

        configuration = {
        'resize_type': args.resize_type,
        'mean_values': args.mean_values,
        'scale_values': args.scale_values,
        'reverse_input_channels': args.reverse_input_channels,
        'path_to_labels': args.labels,
        'confidence_threshold': args.prob_threshold,
        'input_size': args.input_size, # The CTPN specific
        'num_classes': args.num_classes, # The NanoDet and NanoDetPlus specific
        }

        model = DetectionModel.create_model(args.architecture_type, model_adapter, configuration)
        detector_pipeline = AsyncPipeline(model)


        log_adapter_configuration(model_adapter, model, configuration)

        return model, detector_pipeline

def main():
    args = build_argparser().parse_args(rospy.myargv()[1:])
    rospy.init_node('vino_node')
    inference_node(args)
    rospy.spin()

if __name__ == '__main__':
    sys.exit(main() or 0)
