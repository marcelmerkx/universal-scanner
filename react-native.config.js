module.exports = {
  dependency: {
    platforms: {
      android: {
        packageImportPath: 'import com.tflite.TflitePackage;',
        packageInstance: 'new TflitePackage()',
      },
    },
  },
};