#include "dataset.h"
#include "tree.h"
#include "utils.h"
int main() {
	/*FILE *fp = fopen("data.dat", "wb");
	double n;
	for (int i = 0; i < 1000000*100; i++) {
		n = r.next_double();
		fwrite(&n, sizeof(double), 1, fp);
	}*/
	//Dataset ds;
	//ds.readBinary("data.dat", 99);
	// ds.readText("data.txt", 5, TRAIN);
	// print_mat(ds.X, ds.sample_size, ds.feature_size, "feature matrix");
	// print_vec(ds.y, ds.sample_size, "label vector");
	DecisionTreeClassifier *clf = new DecisionTreeClassifier(1, 10);
	clf->train("data2.txt", 100, TEXT);
	clf->export_dotfile("tree.dot");
	double *imp = clf->compute_importance();
	print_vec(imp, 100, "Feature Importance");
	return 0;
}
