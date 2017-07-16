#include "classifier.h"

#include <cmath>

// #include <iostream>

Gaussian::Gaussian():
    Gaussian(0, 1)
{
    // Nothing to do.
}

Gaussian::Gaussian(double mean, double sdev){
    Gaussian::mean = mean;
    Gaussian::sdev = sdev;

    a_ = 1.0 / (sdev * sqrt(2 * M_PI));
    b_ = -0.5 / (sdev * sdev);
}

double Gaussian::operator () (double x) const {
    double d = x - mean;
    return a_ * exp(b_ * d * d);
}

GNB::GNB(const vector<string> &labels):
    classes(labels)
{
    // Nothing to do.
}

void GNB::train(const vector<State> &data, const vector<string> &labels) {
    int m = data.size();
    int n = data[0].size();

    map<string, State> means;
    map<string, State> dev2s;
    for (const string label: classes) {
        means[label].resize(n, 0.0);
        dev2s[label].resize(n, 0.0);
        likelihood[label].resize(n, Gaussian());
        prior[label] = 0.0;
    }

    // Compute per-class sample counts and state attribute sums.
    for (int i = 0; i < m; ++i) {
        auto &sample = data[i];
        auto &label = labels[i];
        auto &sum = means[label];

        prior[label] += 1.0;
        for (int j = 0; j < n; ++j) {
            sum[j] += sample[j];
        }
    }

    // Compute per-class attribute averages.
    for (const string label: classes) {
        auto &count = prior[label];
        auto &mean = means[label];
        for (int j = 0; j < n; ++j) {
            mean[j] /= count;
        }
    }

    // Compute per-class attribute squared deviation sums.
    for (int i = 0; i < m; ++i) {
        auto &sample = data[i];
        auto &label = labels[i];
        auto &mean = means[label];
        auto &dev2 = dev2s[label];
        for (int j = 0; j < n; ++j) {
            double d = sample[j] - mean[j];
            dev2[j] += d * d;
        }
    }

    // Compute per class priors and likelihoods.
    for (const string label: classes) {
        auto &count = prior[label];
        auto &mean = means[label];
        auto &dev2 = dev2s[label];
        auto &cond = likelihood[label];
        for (int j = 0; j < n; ++j) {
            double sdev = sqrt(dev2[j] / count);
            cond[j] = Gaussian(mean[j], sdev);
//             cout << label << '[' << j << "]: likelihood = N(" << mean[j] << ", " << sdev << ")" << endl;
        }

        count /= m; // Turns class count into prior probability

//         cout << label << ": prior = " << count << endl;
    }
}

string GNB::operator () (const State &state) const {
    int n = state.size();
    string c_best;
    double p_best = 0;

    for (const string label: classes) {
        double p = prior.at(label);
        const auto &cond = likelihood.at(label);
        for (int j = 0; j < n; ++j) {
            const auto &f = cond[j];
            p *= f(state[j]);
        }

        if (p_best < p) {
            c_best = label;
            p_best = p;
        }
    }

    return c_best;
}
