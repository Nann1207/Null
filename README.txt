README.txt
----------------------------------------------------------------------------------------------------------------------------------------------------------

Files included:
- Test.py
- Labels.py
- Images (folder)
- ResNet50.py
- NewVGG16.py
- InceptionV3.py


Steps for opening and running the files:

	1.) Create a new folder and move all Python files into it 
		(Test.py, Labels.py, ResNet50.py, NewVGG16.py, InceptionV3.py).

	2.) Move the Images folder into the same folder you created in step 1.

	3.) Open each Python file (except Test.py and Labels.py) and amend the DATA_DIR variable to point to the path of the Images folder.
		For example: DATA_DIR = 'C:\Desktop\Folder\Images'

	4.) Right click inside the folder and open Terminal.

	5.) Type python (name_of_file).py to run the code in Terminal.
		For example: python ResNet50.py

	6.) Repeat step 5 for each python code you want to run.

	7.) Lastly, inside Test.py change the MODEL_PATH and LABELS_PATH path location to where it is automatically generated in the folder.
		For example: MODEL_PATH = 'C:\Desktop\Folder\InceptionV3model.keras'
			     LABELS_PATH = 'C:\Desktop\Folder\Labels.py'

	8.) And repeat step 5 to run Test.py