/*
 * Copyright [2017] Wikimedia Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.o19s.es.ltr.ranker.dectree;

import com.o19s.es.ltr.ranker.DenseFeatureVector;
import com.o19s.es.ltr.ranker.DenseFeatureVectorDouble;
import com.o19s.es.ltr.ranker.DenseLtrRanker;
import com.o19s.es.ltr.ranker.normalizer.Normalizer;
import org.apache.lucene.util.Accountable;
import org.apache.lucene.util.RamUsageEstimator;

import java.util.Objects;

/**
 * Naive implementation of additive decision tree.
 * May be slow when the number of trees and tree complexity if high comparatively to the number of features.
 */
public class NaiveAdditiveDecisionTreeDouble extends DenseLtrRanker implements Accountable {
    private static final long BASE_RAM_USED = RamUsageEstimator.shallowSizeOfInstance(SplitDouble.class);

    private final NodeDouble[] trees;
    private final double[] weights;
    private final int modelSize;
    private final Normalizer normalizer;

    /**
     * TODO: Constructor for these classes are strict and not really
     * designed for a fluent building process. We might consider
     * changing this according to model parsers we implement.
     *
     * @param trees an array of trees
     * @param weights the respective weights
     * @param modelSize the modelSize in number of feature used
     * @param normalizer class to perform any normalization on model score
     */
    public NaiveAdditiveDecisionTreeDouble(NodeDouble[] trees, double[] weights, int modelSize, Normalizer normalizer) {
        assert trees.length == weights.length;
        this.trees = trees;
        this.weights = weights;
        this.modelSize = modelSize;
        this.normalizer = normalizer;
    }

    @Override
    public String name() {
        return "naive_additive_decision_tree";
    }

    @Override
    protected float score(DenseFeatureVector vector) {
        assert(false);
       return 0; 
    }

    @Override
    protected double score(DenseFeatureVectorDouble vector) {
        double sum = 0;
        double[] scores = vector.scores;
        for (int i = 0; i < trees.length; i++) {
            sum += weights[i]*trees[i].eval(scores);
        }
        return sum;
//        return normalizer.normalize(sum);
    }

    @Override
    protected int size() {
        return modelSize;
    }

    /**
     * Return the memory usage of this object in bytes. Negative values are illegal.
     */
    @Override
    public long ramBytesUsed() {
        return BASE_RAM_USED + RamUsageEstimator.sizeOf(weights)
                + RamUsageEstimator.sizeOf(trees);
    }

    public interface Node extends Accountable {
         boolean isLeaf();
         float eval(float[] scores);
    }

    public interface NodeDouble extends Accountable {
        boolean isLeaf();
        double eval(double[] scores);
    }

    public static class SplitDouble implements NodeDouble {
        private static final long BASE_RAM_USED = RamUsageEstimator.shallowSizeOfInstance(SplitDouble.class);
        private final NodeDouble left;
        private final NodeDouble right;
        private final int feature;
        private final double threshold;

        public SplitDouble(NodeDouble left, NodeDouble right, int feature, double threshold) {
            this.left = Objects.requireNonNull(left);
            this.right = Objects.requireNonNull(right);
            this.feature = feature;
            this.threshold = threshold;
        }

        @Override
        public boolean isLeaf() {
            return false;
        }

        @Override
        public double eval(double[] scores) {
            NodeDouble n = this;
            while (!n.isLeaf()) {
                assert n instanceof SplitDouble;
                SplitDouble s = (SplitDouble) n;
                if (s.threshold > scores[s.feature]) {
                    n = s.left;
                } else {
                    n = s.right;
                }
            }
            assert n instanceof LeafDouble;
            return n.eval(scores);
        }

        /**
         * Return the memory usage of this object in bytes. Negative values are illegal.
         */
        @Override
        public long ramBytesUsed() {
            return BASE_RAM_USED + left.ramBytesUsed() + right.ramBytesUsed();
        }
    }

    public static class LeafDouble implements NodeDouble {
        private static final long BASE_RAM_USED = RamUsageEstimator.shallowSizeOfInstance(LeafDouble.class);

        private final double output;

        public LeafDouble(double output) {
            this.output = output;
        }

        @Override
        public boolean isLeaf() {
            return true;
        }

        @Override
        public double eval(double[] scores) {
            return output;
        }

        /**
         * Return the memory usage of this object in bytes. Negative values are illegal.
         */
        @Override
        public long ramBytesUsed() {
            return BASE_RAM_USED;
        }
    }
}
