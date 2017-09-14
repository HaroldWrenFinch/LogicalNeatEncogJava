package com.haroldwren.machine.logical.neat;

import org.encog.Encog;
import org.encog.ml.CalculateScore;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.ea.train.EvolutionaryAlgorithm;
import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.neat.NEATPopulation;
import org.encog.neural.neat.NEATUtil;
import org.encog.neural.networks.training.TrainingSetScore;
import org.encog.util.simple.EncogUtility;

public class NeatOrEncog {
    public static double OR_INPUT[][] = {
            {0.0, 0.0},
            {1.0, 0.0},
            {0.0, 1.0},
            {1.0, 1.0}
    };

    public static double OR_OUTPUT[][] = {
            {0.0},
            {1.0},
            {1.0},
            {1.0}
    };

    public static void main(String[] args) {
        MLDataSet trainingSet = new BasicMLDataSet(OR_INPUT, OR_OUTPUT);
        NEATPopulation pop = new NEATPopulation(2,1,1000);
        pop.setInitialConnectionDensity(1.0);// not required, but speeds training
        pop.reset();

        CalculateScore score = new TrainingSetScore(trainingSet);

        final EvolutionaryAlgorithm mlTrain = NEATUtil.constructNEATTrainer(pop,score);

        int iteration = 1;
        do {
            mlTrain.iteration();
            System.out.println("Iteration: " + iteration + ", Error: " + mlTrain.getError() + ", Species: " + pop.getSpecies().size());
            iteration++;
        } while (mlTrain.getError() > 0.001);

        NEATNetwork network = (NEATNetwork)mlTrain.getCODEC().decode(mlTrain.getBestGenome());

        System.out.println("Neural Network Results:");
        EncogUtility.evaluate(network, trainingSet);

        Encog.getInstance().shutdown();
    }
}


