import pandas as pd
def load_data(path):
    df=pd.read_csv(path)
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
    return df