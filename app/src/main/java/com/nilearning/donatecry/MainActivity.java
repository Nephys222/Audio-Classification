package com.nilearning.donatecry;

import android.media.AudioRecord;
import android.os.Bundle;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.util.Log;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.support.audio.TensorAudio;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.task.audio.classifier.AudioClassifier;
import org.tensorflow.lite.task.audio.classifier.Classifications;

import java.util.Arrays;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.stream.Collectors;

public class MainActivity extends AppCompatActivity {

    // TODO 1: define your model name
    String modelPath = "donate_a_cry.tflite";

    float probThresholdMain = 0.2f;
    float probThresholdSecond = 0.5f;

    TextView resultText;
    TextView specsText;

    private final ActivityResultLauncher<String> requestVoicePermission =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    startRecording();
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        resultText = findViewById(R.id.output);
        specsText = findViewById(R.id.audio_recorder_specs);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            startRecording();
        } else {
            requestVoicePermission.launch(Manifest.permission.RECORD_AUDIO);
        }
    }

    private void startRecording() {
        try {
            AudioClassifier classifier = AudioClassifier.createFromFile(this, modelPath);
            TensorAudio tensor = classifier.createInputTensorAudio();
            TensorAudio.TensorAudioFormat format = classifier.getRequiredTensorAudioFormat();

            String recorderSpecs = "Number Of Channels: " + format.getChannels() + "\n" +
                    "Sample Rate: " + format.getSampleRate();
            specsText.setText(recorderSpecs);

            AudioRecord record = classifier.createAudioRecord();
            record.startRecording();

            new Timer().schedule(new TimerTask() {
                @Override
                public void run() {
                    int numberOfSamples = tensor.load(record);
                    List<Classifications> output = classifier.classify(tensor);

                    // TODO 2: Check if it's a target sound.
                    List<Category> filteredModelOutput = output.get(0).getCategories().stream()
                            .filter(category -> {
                                List<String> targetLabels = Arrays.asList(
                                        "Child speech, kid speaking", "Screaming", "Children shouting", "Baby laughter", "Chuckle, chortle",
                                        "Crying, sobbing", "Baby cry, infant cry", "Child singing", "Burping, eructation");
                                return targetLabels.contains(category.getLabel()) && category.getScore() > probThresholdMain;
                            })
                            .collect(Collectors.toList());

                    // TODO 3: given there's a target sound, which one is it?
                    if (!filteredModelOutput.isEmpty()) {
                        filteredModelOutput = output.get(1).getCategories().stream()
                                .filter(category -> {
                                    Log.i("Yamnet", "EC Score " + category.getScore() + "; L: " + category.getLabel());
                                    return category.getScore() > probThresholdSecond;
                                })
                                .collect(Collectors.toList());
                    }

                    String outputStr = filteredModelOutput.stream()
                            .sorted((c1, c2) -> Float.compare(c2.getScore(), c1.getScore()))
                            .map(category -> category.getLabel() + " -> " + category.getScore() + " ")
                            .collect(Collectors.joining("\n"));

                    if (!outputStr.isEmpty()) {
                        runOnUiThread(() -> resultText.setText(outputStr));
                    }
                }
            }, 1, 500);

        } catch (Exception e) {
            Log.e("AudioClassification", "Error classifying audio: " + e.getMessage());
        }
    }
}
