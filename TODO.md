# Files to visualize interactive charts

The folders store the results from umap and t-SNE to visualize the distributions of all datasests (KuHar, MotionSense, UCIHAR, WISDM, and ExtraSensory).

The results are organized in:

- *Charts*: There are results in svg format;
- *htmlFile*: There are results in html format;
- *jsonFile*: There are results in json format;
- *pngFile*: There are results in png format;

Each file is named as follows:

**Domain**_data_**model**_label_**labels_select**_**format**

where,

- Domain = Time or Frequency
- model = Umap or t-SNE

- labes_select = 
1. DataSet: the label is the dataset name
2. Activity_Dataset: the labels is the activity with dataset name
3. Activity: the label is just the activity
3. Activity from standard activity code: the label is the dataset name
4. Activity from standard activity code and user: the label is the dataset name and it show the user code
5. user_on activity from standard activity code: the label is the activity and it show the user code