
[ ] run the better detection model (v10 training), but based on black/white

[ ] put the "current best" model in .env and make scripts use that. improves re-usability!

[ ] deal with some artefacts: 
``` 
type: 'code_' + classNames[bestClass]
model: 'unified-detection-v7.tflite'
```

[ ] header file ImageData.h has logic; should go to .cpp file. header will stay header.

[ ] naming: letterboxedCrop definition for a right-padded vertical container sounds like a misnaming

[ ] definition of "stages" "phases" and "steps" for our logs. Make it structured , maybe coded so we know what we're looking at in the adb logs when working verbosely

[ ] nativeProcessFrameWithData rename width and height to frameWidth and frameHeight from hereon.

[ ] ide errors 

[ ] ios

