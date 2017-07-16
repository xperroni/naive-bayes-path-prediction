#include "classifier.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

vector<State> Load_State(string file_name) {
    ifstream in_state_(file_name.c_str(), ifstream::in);
    vector<State> states;

    string line;
    while (getline(in_state_, line)) {
        states.emplace_back();
        State &state = states.back();
        state.resize(4);

        istringstream iss(line);
        for (int i = 0; i < 4; ++i) {
            iss >> state[i];
            iss.get();
        }
    }

    return states;
}

vector<string> Load_Label(string file_name)
{
    ifstream in_label_(file_name.c_str(), ifstream::in);
    vector< string > label_out;
    string line;
    while (getline(in_label_, line))
    {
    	istringstream iss(line);
    	string label;
	    iss >> label;

	    label_out.push_back(label);
    }
    return label_out;

}

int main() {

    vector<State> X_train = Load_State("./data/train_states.txt");
    vector<State> X_test  = Load_State("./data/test_states.txt");
    vector< string > Y_train  = Load_Label("./data/train_labels.txt");
    vector< string > Y_test   = Load_Label("./data/test_labels.txt");

	cout << "X_train number of elements " << X_train.size() << endl;
	cout << "X_train element size " << X_train[0].size() << endl;
	cout << "Y_train number of elements " << Y_train.size() << endl;

	GNB gnb({"keep", "left", "right"});

	gnb.train(X_train, Y_train);

	cout << "X_test number of elements " << X_test.size() << endl;
	cout << "X_test element size " << X_test[0].size() << endl;
	cout << "Y_test number of elements " << Y_test.size() << endl;

	int score = 0;
	for(int i = 0, n = X_test.size(); i < n; i++)
	{
		vector<double> coords = X_test[i];
		string predicted = gnb(coords);
		if(predicted.compare(Y_test[i]) == 0)
		{
			score += 1;
		}
	}

	float fraction_correct = float(score) / Y_test.size();
	cout << "You got " << (100*fraction_correct) << " correct" << endl;

	return 0;
}
