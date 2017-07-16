#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <map>
#include <vector>

using namespace std;

typedef vector<double> State;

struct Gaussian {
    /** @brief Peak of the curve. */
    double mean;

    /** @brief Spread of the curve. */
    double sdev;

    /**
     * @brief Create a new Gaussian of mean `0` and standard deviation `1`.
     */
    Gaussian();

    /**
     * @brief Create a new Gaussian of given mean and standard deviation.
     */
    Gaussian(double mean, double sdev);

    /**
     * @brief Compute the value of the Gaussian curve for the given input.
     */
    double operator () (double x) const;

private:
    /** @brief Product of all constant terms in the Gaussian function outside the exponential. */
    double a_;

    /** @brief Product of all constant terms in the Gaussian function inside the exponential. */
    double b_;
};

struct GNB {
    /** @brief Vector of class labels. */
    vector<string> classes;

    /** @brief Prior probabilities by class. */
    map<string, double> prior;

    /** @brief Conditional likelihoods by class. */
    map<string, vector<Gaussian>> likelihood;

    /**
     * @brief Create a new Gaussian naive Bayes classifier for the given classes.
     *
     * @param labels Vector of class labels.
     */
    GNB(const vector<string> &labels);

    /**
     * @brief Train the classifier on the given data.
     */
    void train(const vector<State> &data, const vector<string> &labels);

    /**
     * @brief Return the label of the class closest to the given state.
     */
    string operator () (const State &state) const;
};

#endif
