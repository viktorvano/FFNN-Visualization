package FFNN;

import java.util.ArrayList;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.layout.Pane;
import javafx.stage.Stage;
import javafx.util.Duration;

import static FFNN.GeneralFunctions.*;

public class FFNN extends Application
{
    public static void main(String[] args)
    {
        launch(args);
    }

    private Pane pane;
    private Button[] btnInputs;
    private Button[] btnOutputs;
    private Label[] lblOutputs;
    private float[] inputColor;
    private float[] outputColor;
    private final int width = 1600;
    private final int height = 900;
    private Timeline timelineNeuralNetRun;
    private Timeline timelineNeuralNetLoading;
    private NeuralNetwork myNet;
    private NeuralNetObjects neuralNetObjects;
    private boolean netLoading = true;
    private Button btnRun;
    private ArrayList<ArrayList<Button>> btnHidden = new ArrayList<>();
    private long totalTime = 0;
    private int cycles = 0;

    @Override
    public void start(Stage stage)
    {
        pane = new Pane();
        Scene scene = new Scene(pane, width, height);

        stage.setTitle("FFNN Visualization");
        stage.setScene(scene);
        stage.show();
        stage.setMaxWidth(stage.getWidth());
        stage.setMinWidth(stage.getWidth());
        stage.setMaxHeight(stage.getHeight());
        stage.setMinHeight(stage.getHeight());
        stage.setResizable(false);

        try
        {
            Image icon = new Image(getClass().getResourceAsStream("../images/neural-network-icon.jpg"));
            stage.getIcons().add(icon);
            System.out.println("Icon loaded from IDE...");
        }catch(Exception e)
        {
            try
            {
                Image icon = new Image("images/neural-network-icon.jpg");
                stage.getIcons().add(icon);
                System.out.println("Icon loaded from exported JAR...");
            }catch(Exception e1)
            {
                System.out.println("Icon failed to load...");
            }

        }

        String fileSeparator = System.getProperty("file.separator");
        String topologyFilePath = "res" + fileSeparator + "topology.txt";
        String trainingFilePath = "res" + fileSeparator + "training.txt";
        String weightsFilePath = "res" + fileSeparator + "weights.txt";
        neuralNetObjects = new NeuralNetObjects(topologyFilePath, trainingFilePath, weightsFilePath,0.1f, 0.5f, 0.0003f, 50000);
        myNet = new NeuralNetwork(neuralNetObjects);
        if (neuralNetObjects.topology.get(0) != 9)
        {
            System.out.println("Topology ERROR:\nNeural network must have 9 inputs.");
            return;
        }

        if (neuralNetObjects.topology.get(neuralNetObjects.topology.size()-1) != 4)
        {
            System.out.println("Topology ERROR:\nNeural network must have 4 outputs.");
            return;
        }

        btnInputs =  new Button[9];
        inputColor = new float[9];
        for(int i=0; i<btnInputs.length; i++)
        {
            btnInputs[i] = new Button();
            btnInputs[i].setPrefSize(50, 50);
            inputColor[i] = 0.0f;
            btnInputs[i].setStyle(colorStyle(inputColor[i]));
            btnInputs[i].setText(formatFloatToString4(inputColor[i]));
            if(i<3)
            {
                btnInputs[i].setLayoutY(340);
                btnInputs[i].setLayoutX(100+60*i);
            }else if(i>=3 && i<6)
            {
                btnInputs[i].setLayoutY(400);
                btnInputs[i].setLayoutX(100+60*(i-3));
            }else if(i>=6 && i<9)
            {
                btnInputs[i].setLayoutY(460);
                btnInputs[i].setLayoutX(100+60*(i-6));
            }
        }

        btnOutputs =  new Button[4];
        outputColor = new float[4];
        lblOutputs = new Label[4];
        for(int i=0; i<btnOutputs.length; i++)
        {
            btnOutputs[i] = new Button();
            btnOutputs[i].setPrefSize(70, 70);
            outputColor[i] = 0.0f;
            btnOutputs[i].setStyle(colorStyle(outputColor[i]));
            btnOutputs[i].setText(formatFloatToString4(outputColor[i]));

            btnOutputs[i].setLayoutX(1420);
            btnOutputs[i].setLayoutY(180+150*i);

            lblOutputs[i] = new Label();
            lblOutputs[i].setLayoutX(1500);
            lblOutputs[i].setLayoutY(180+150*i);
        }

        lblOutputs[0].setText("0  1  0\n1  0  1\n0  1  0");
        lblOutputs[1].setText("0  0  0\n1  1  1\n0  0  0");
        lblOutputs[2].setText("1  0  1\n0  1  0\n1  0  1");
        lblOutputs[3].setText("0  1  0\n1  1  1\n0  1  0");

        btnRun = new Button("Run");
        btnRun.setPrefSize(80, 40);
        btnRun.setLayoutX(50);
        btnRun.setLayoutY(50);
        btnRun.setDisable(true);
        btnRun.setOnAction(event-> {
            if(!netLoading)
            {
                btnRun.setDisable(true);
                timelineNeuralNetRun.play();
            }
        });

        pane.getChildren().addAll(btnInputs);
        pane.getChildren().addAll(btnOutputs);
        pane.getChildren().addAll(lblOutputs);
        pane.getChildren().add(btnRun);

        timelineNeuralNetRun = new Timeline(new KeyFrame(Duration.millis(20), event -> runCycle()));
        timelineNeuralNetRun.setCycleCount(Timeline.INDEFINITE);

        int x_range, y_range;
        x_range = 900/(neuralNetObjects.topology.size() - 3);
        for(int x=1; x< neuralNetObjects.topology.size()-1; x++)//X = 900 pix range
        {
            btnHidden.add(new ArrayList<>());
            for(int y=0; y<neuralNetObjects.topology.get(x); y++)//Y = 750 pix range
            {
                y_range = 750/neuralNetObjects.topology.get(x);
                btnHidden.get(x-1).add(new Button("0"));
                btnHidden.get(x-1).get(y).setLayoutX(350+(x-1)*x_range);
                btnHidden.get(x-1).get(y).setLayoutY(100+y*y_range);
                btnHidden.get(x-1).get(y).setPrefSize(70, 40);
                btnHidden.get(x-1).get(y).setStyle("-fx-background-color: #000000;");
                pane.getChildren().add(btnHidden.get(x-1).get(y));
            }
        }

        timelineNeuralNetLoading  = new Timeline(new KeyFrame(Duration.millis(250), event -> {
            if(!myNet.isNetLoading())
            {
                netLoading = false;
                timelineNeuralNetRun.play();
                timelineNeuralNetLoading.stop();
            }
        }));
        timelineNeuralNetLoading.setCycleCount(Timeline.INDEFINITE);
        timelineNeuralNetLoading.play();
    }

    private void runCycle()
    {
        //trainingPass++;
        //System.out.println("Run: " + trainingPass);

        //Get new input data and feed it forward:
        //Make sure that your input data are the same size as InputNodes
        neuralNetObjects.input.clear();
        for(int i = 0; i < neuralNetObjects.inputNodes; i++)
        {
            neuralNetObjects.input.add((float)(Math.round(Math.random())));
            inputColor[i] = neuralNetObjects.input.get(neuralNetObjects.input.size()-1);
            btnInputs[i].setStyle(colorStyle(inputColor[i]));
            btnInputs[i].setText(formatFloatToString4(inputColor[i]));
        }
        showVectorValues("Inputs:", neuralNetObjects.input);
        long start, end;
        cycles++;
        start = System.nanoTime();
        myNet.feedForward(neuralNetObjects.input);
        end = System.nanoTime();
        totalTime += (end - start);

        for(int x=1; x< neuralNetObjects.topology.size()-1; x++)//X = 900 pix range
        {
            btnHidden.add(new ArrayList<>());
            for(int y=0; y<neuralNetObjects.topology.get(x); y++)//Y = 750 pix range
            {
                float color = myNet.getNeuronOutput(x, y);
                btnHidden.get(x-1).get(y).setText(formatFloatToString4(color));
                btnHidden.get(x-1).get(y).setStyle(colorStyle(color));
            }
        }

        // Collect the net's actual results:
        myNet.getResults(neuralNetObjects.result);
        showVectorValues("Outputs: ", neuralNetObjects.result);

        for(int i = 0; i < neuralNetObjects.outputNodes; i++)
        {
            outputColor[i] = neuralNetObjects.result.get(i);
            btnOutputs[i].setStyle(colorStyle(outputColor[i]));
            btnOutputs[i].setText(formatFloatToString4(outputColor[i]));

            if(outputColor[i]>0.5)
            {
                timelineNeuralNetRun.stop();
                btnRun.setDisable(false);
                double averageTimeMillis = ((double) totalTime/1000.0) / (double) cycles;
                System.out.println("Average Feed Forward time [us]: " + averageTimeMillis);
                System.out.println("Total Cycles: " + cycles);
            }
        }
    }
}
