import numpy as np
import pandas as pd
from data_cleaning_pipeline import pipeline



df=pd.read_csv('C://Users//koushik//Desktop//project//data//flat_dataset//flat_data.csv')
df.columns = [col.split('.')[-1] for col in df.columns]
df = df[df['SummaryCategories'] != 'Folder'] #Folder columns
df.drop(['fileversion','ShortName','process','testCaseShoortName','State','RequirementID','testcaseId',
         'refId','refMode','modifiedBy','createdBy','modifiedBy','priority','externalId',
         'externalScriptName','createdDate','modifiedDate','automatedBehavioronError','tocomment',
         'fromcomment','displayOnTree','displayOnTable','path','documentId','status','version',
         'updateversion','createdate','modifiedate','parentId','testcasePTCModifiedDate',
         'summaryCategoriesModifyflag','descriptionModifyflag','testStepsModifyflag','testSteps',
         'preconditionModifyflag','postconditionModifyflag','expectedResultsModifyflag',
         'parameterPtclistModifyflag','StateModifyflag','priorityModifyflag','tocommentModifyflag',
         'statusflag','traceabiltyMofifyflag','mitescriptflag','checkBoxflag','testcaseenable',
         'testScriptEditFlag','testScriptName','miteXmlData','deleterelations','insertLocation',
         'SummaryCategories','expectedResults',
         'precondition','postcondition',
         'ParameterValues', #only one row have this
         'RelationShips'],axis=1,inplace=True)

#intermediate data
df.to_csv("C://Users//koushik//Desktop//project//data//flat_dataset//intermediate_dataset.csv", index=False, encoding='utf-8')


df=pipeline(df)

df.to_csv("C://Users//koushik//Desktop//project//data//flat_dataset//final_dataset.csv", index=False, encoding='utf-8')



