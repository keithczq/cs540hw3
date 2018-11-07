import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.*;
import java.lang.Math;

/**
 * Fill in the implementation details of the class DecisionTree using this file. Any methods or
 * secondary classes that you want are fine but we will only interact with those methods in the
 * DecisionTree framework.
 * 
 * You must add code for the 1 member and 4 methods specified below.
 * 
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {
  private DecTreeNode root;
  //ordered list of class labels
  private List<String> labels; 
  //ordered list of attributes
  private List<String> attributes; 
  //map to ordered discrete values taken by attributes
  private Map<String, List<String>> attributeValues; 
  //map for getting the index
  private HashMap<String,Integer> label_inv;
  private HashMap<String,Integer> attr_inv;
  
  /**
   * Answers static questions about decision trees.
   */
  DecisionTreeImpl() {
    // no code necessary this is void purposefully
  }

  /**
   * Build a decision tree given only a training set.
   * 
   * @param train: the training set
   */
  DecisionTreeImpl(DataSet train) {

    this.labels = train.labels;
    this.attributes = train.attributes;
    this.attributeValues = train.attributeValues;
    // Get the list of instances via train.instances
    // You should write a recursive helper function to build the tree
    //
    // this.labels contains the possible labels for an instance
    // this.attributes contains the whole set of attribute names
    // train.instances contains the list of instances
    
    root = this.buildTree(train.instances, this.attributes, this.majorityLabel(train.instances), null);
  }
  
  /*HELPER METHOD
   * Builds the tree based on current train */
  public DecTreeNode buildTree(List<Instance> instances, List<String> attributes, String defaultLabel, String parentValue) {
	  if (instances.isEmpty()) {
		  return new DecTreeNode(defaultLabel, null, parentValue, true);							
	  }
	  if (this.sameLabel(instances)) {
		  return new DecTreeNode(this.majorityLabel(instances), null, parentValue, true);				
	  }
	  if (attributes.isEmpty()) {
		  return new DecTreeNode(this.majorityLabel(instances), null, parentValue, true);				
	  }
	  
	  String q = this.best_attribute(instances, attributes);
	  //check if q is null which means all info gain is negative and hence do not take 
	  if (q == null) {
		  return new DecTreeNode(this.majorityLabel(instances), null, parentValue, true);
	  }
	  
	  DecTreeNode tree = new DecTreeNode(null, q, parentValue, false); 
	  
	  List<String> attrTypes = attributeValues.get(q); //obtain the different attribute types for q
	  
	//Remove attribute form attributes list
	  List<String> lessAttributes = new ArrayList<String>(attributes);
	  lessAttributes.remove(q);
	  
	  //Loop through each value v of q
	  for (int j = 0; j < attrTypes.size(); j++) {
		  String currAttrValue = attrTypes.get(j); //name of current attribute value subtree being expanded
		  //Partition instances using possible values of attribute q
		  List<Instance> partInst = new ArrayList<Instance>(); 										
		  for (int k = 0; k < instances.size(); k++) {
			  int index = this.getAttributeIndex(q); //attribute index of the current attribute we are finding
			  String currInstAttr = instances.get(k).attributes.get(index); //name of current instance's attribute
			  //Check if the instance attribute matches the one we are finding for
			  if (currInstAttr.equals(currAttrValue)) {
				  //Since they match, add the current instance to the new partitioned instance
				  partInst.add(instances.get(k));
			  }
		  }
		  
		  
		  
		  //Create subtree
		  DecTreeNode subTree = buildTree(partInst, lessAttributes, partInst.size()==0? defaultLabel:this.majorityLabel(partInst), currAttrValue); 
		  //Add subtree to our original tree
		  tree.addChild(subTree);
	  }
	 
	  return tree;
  }
  
  /*HELPER METHOD
   * Calculates which is the best attribute
   * */
  public String best_attribute(List<Instance> instances, List<String> attributes) {
	  String best = null;
	  double bestGain = -1.0;
	  //Iterate through attributes list to find best attribute
	  for (int i = 0; i < attributes.size(); i++) {
		  double currGain = this.InfoGain(instances, attributes.get(i));
		  if (currGain > bestGain) {
			  //update best gain value and reference
			  bestGain = currGain;
			  best = attributes.get(i);
		  }
	  }
	  return best;
  }

  boolean sameLabel(List<Instance> instances){
      // Suggested helper function
      // returns if all the instances have the same label
      // labels are in instances.get(i).label
	  String firstLabel = instances.get(0).label;
	  for (int i = 1; i < instances.size(); i++) {
		  String currLabel = instances.get(i).label;
		  if (!currLabel.equals(firstLabel)) {
			  return false;
		  }
	  }
      return true;
  }
  String majorityLabel(List<Instance> instances){
      // Suggested helper function
      // returns the majority label of a list of examples
	  
	  //Obtain label string of first label
	  String aLabel = this.labels.get(0);										 
	  String bLabel = this.labels.get(1);
	  
	  //Since binary classification, only need to keep track of one and subtract from total to get the other
	  double aCount = 0.0; //counter to keep track of count of the first label
	  //Calculate probability the labels
	  for (int i = 0; i < instances.size(); i++) {
		  if (instances.get(i).label.equals(aLabel)) {
			  aCount++;
		  }
	  }
	  //Check whether a or b label is more
	  if (aCount >= ((double)instances.size() / 2.0)) {
		  //since alabel occured first or is the majority, return it
		  return aLabel;
	  }
	  else {
		  return bLabel; //line is reached if a is not majority, hence return b label
	  }
  }
  double entropy(List<Instance> instances){
      // Suggested helper function
	  //Math.log(n) / Math.log(2), with n being the number you're trying to get the base 2 log value from
	  if (instances.size() == 0) {
		  return 0.0;
	  }
	  //Obtain label string of first label
	  String aLabel = this.labels.get(0);										
	  
	  //Since binary classification, only need to keep track of one and subtract from total to get the other
	  int aCount = 0; //counter to keep track of count of the first label
	  //Calculate probability the labels
	  for (int i = 0; i < instances.size(); i++) {
		  if (instances.get(i).label.equals(aLabel)) {
			  aCount++;
		  }
	  }
	  //Calculate probability of both labels
	  double aProb = (double) aCount / instances.size();
	  double bProb = 1 - aProb; 
	  
	  //Calculate entropy of each label
	  double aEntropy = 0.0;
	  if (aCount > 0 && aProb != 1) { 			
		  aEntropy = -aProb*(Math.log(aProb) / Math.log(2));
	  }
	  
	  double bEntropy = 0.0;
	  int bCount = instances.size() - aCount;
	  if (bCount > 0 && bProb != 1) { 			
		  bEntropy = -bProb*(Math.log(bProb) / Math.log(2));
	  }
      return aEntropy + bEntropy;
  }
  double conditionalEntropy(List<Instance> instances, String attr){
	  if (instances.size() == 0) {
		  return 0.0;
	  }
	  
//	  //NEW
//	  double entropy = 0.0;
//	  int attrIndex = this.getAttributeIndex(attr);
//	  List<Instance> condAttributes = new ArrayList<Instance>(); //create a list of instances that match condition
//	  for (int i = 0; i < instances.size(); i++) {
//		  String instAttr = instances.get(i).attributes.get(attrIndex);
//		  if (instAttr.equals(attr)) {
//			  condAttributes.add(instances.get(i));
//		  }
//	  }
//	  double attrProb = condAttributes.size() / attributes.size();
//	  for (int j = 0; j < condAttributes.size(); j++) {
//		  entropy += this.entropy(condAttributes);
//	  }
	  
	  //OLD
	  ArrayList<String> attributes = new ArrayList<String>();
	  int attrIndex = this.getAttributeIndex(attr);							
	  //Loop through to get the number of different attributes 
	  for (int i = 0; i < instances.size(); i++) {
		  String curr = instances.get(i).attributes.get(attrIndex);
		  if (!attributes.contains(curr)) {
			  attributes.add(curr);
		  }
	  }
	  //At the end of loop, attributes list will contain the different type of attribute values
	  
	  double entropy = 0.0; //to store total entropy
	  for (int j = 0; j < attributes.size(); j++) {															
		  String currAttr = attributes.get(j); //attribute value to be checked for occurence in instances
		  String aLabel = this.labels.get(0);
		  double aCount = 0; //current count of attribute based on first label
		  double bCount = 0; //current count of attribute based on second label 
		  //for loop to obtain the counts of each label for current attribute
		  for (int k = 0; k < instances.size(); k++) {
			  //attribute value of current instance node we are checking
			  String instAttr = instances.get(k).attributes.get(attrIndex); 
			  if (instAttr.equals(currAttr)) {

				  if (instances.get(k).label.equals(aLabel)) {
					  aCount++; //has attribute value and first label value
				  }
				  else {
					  bCount++; //has attribute value and second label value
				  }
			  }
		  }
		  
		  double attrProb = (double)(aCount + bCount) / instances.size();
		  //Calculate entropy of each label and add to total entropy if probability is not zero
		  double aProb = 0.0;
		  if (aCount > 0) { 	
			  aProb = (double) aCount / (aCount + bCount);
			  if (aProb != 1) {
				  entropy += attrProb *-aProb*(Math.log(aProb) / Math.log(2));  
			  }
		  }
//		  double bAttrProb = 1 - aAttrProb;
		  double bProb = 0.0;
		  if (bCount > 0) { 			
			  bProb = (double) bCount / (aCount + bCount);
			  if (bProb != 1) {
				  entropy += attrProb * (-bProb*(Math.log(bProb) / Math.log(2)));  
			  }  
		  }
		 
	  }

      return entropy;
  }
  double InfoGain(List<Instance> instances, String attr){
      // Suggested helper function
      // returns the info gain of a list of examples, given the attribute attr
      return entropy(instances) - conditionalEntropy(instances,attr);
  }
  @Override
  public String classify(Instance instance) {
      // The tree is already built, when this function is called
      // this.root will contain the learnt decision tree.
      // write a recusive helper function, to return the predicted label of instance
	  DecTreeNode curr = root; //to store current node that we have in
	  curr = classifyHelper(instance, curr); //update current node to be leaf node with highest gain in tree
	  
    return curr.label; 
  }
  
  /* HELPER METHOD FOR CLASSIFY
   * Recursively calls and obtains the child that matches the attribute values of instance
   */
  public DecTreeNode classifyHelper(Instance instance, DecTreeNode curr) {
	  //check root attribute
	  //check value of instance for attribute
	  //pick child whose parent attribute value == instance attribute value
	  //update node with found child
	  //call next iteration
	  
	  if (curr.terminal) {
		  return curr;
	  }
	  DecTreeNode child = null; //to store node that has matching attribute value as instance
	  String currAttr = curr.attribute; //store current node's attribute
	  int currAttrIndex = this.getAttributeIndex(currAttr);						 
	  String instAttrVal = instance.attributes.get(currAttrIndex); //obtain value of instance's attribute value that is split on
	  for (int i = 0; i < curr.children.size(); i++) {
		  String currAttrVal = curr.children.get(i).parentAttributeValue; //curent node's attribute value
		  if (instAttrVal.equals(currAttrVal)) {
			  child = curr.children.get(i); //store reference to the correct node with attribute value
			  break;
		  }
	  }
	  
	  return this.classifyHelper(instance, child); //recursively call classify helper until terminal node is reached
  }
  
  @Override
  public void rootInfoGain(DataSet train) {
    this.labels = train.labels;
    this.attributes = train.attributes;
    this.attributeValues = train.attributeValues; 		
    // Print the Info Gain for using each attribute at the root node
    // The decision tree may not exist when this funcion is called.
    // But you just need to calculate the info gain with each attribute,
    // on the entire training set.
    for (int i = 0; i < train.attributes.size(); i++) {
    	String currAttr = train.attributes.get(i); //name of the current attribute
    	double currAttrValue = this.InfoGain(train.instances, currAttr);
    	System.out.print(currAttr);
    	System.out.format(" %.5f\n", currAttrValue);
	 }
    
    
  }
  @Override
  public void printAccuracy(DataSet test) {
    // Print the accuracy on the test set.
    // The tree is already built, when this function is called
    // You need to call function classify, and compare the predicted labels.
    // List of instances: test.instances 
    // getting the real label: test.instances.get(i).label

	  double correct = 0; //store number that was correctly classified
	  for (int i = 0; i < test.instances.size(); i++) {
		  if (this.classify(test.instances.get(i)).equals(test.instances.get(i).label)) { //FIXME correct??
			  correct++;
		  }
	  }
	  double acc = correct / test.instances.size();
	  System.out.format("%.5f\n", acc);
    return;
  }
  
  @Override
  /**
   * Print the decision tree in the specified format
   * Do not modify
   */
  public void print() {

    printTreeNode(root, null, 0);
  }

  /**
   * Prints the subtree of the node with each line prefixed by 4 * k spaces.
   * Do not modify
   */
  public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < k; i++) {
      sb.append("    ");
    }
    String value;
    if (parent == null) {
      value = "ROOT";
    } else {
      int attributeValueIndex = this.getAttributeValueIndex(parent.attribute, p.parentAttributeValue);
      value = attributeValues.get(parent.attribute).get(attributeValueIndex);
    }
    sb.append(value);
    if (p.terminal) {
      sb.append(" (" + p.label + ")");
      System.out.println(sb.toString());
    } else {
      sb.append(" {" + p.attribute + "?}");
      System.out.println(sb.toString());
      for (DecTreeNode child : p.children) {
        printTreeNode(child, p, k + 1);
      }
    }
  }

  /**
   * Helper function to get the index of the label in labels list
   */
  private int getLabelIndex(String label) {
    if(label_inv == null){
        this.label_inv = new HashMap<String,Integer>();
        for(int i=0; i < labels.size();i++)
        {
            label_inv.put(labels.get(i),i);
        }
    }
    return label_inv.get(label);
  }
 
  /**
   * Helper function to get the index of the attribute in attributes list
   */
  private int getAttributeIndex(String attr) {
    if(attr_inv == null)
    {
        this.attr_inv = new HashMap<String,Integer>();
        for(int i=0; i < attributes.size();i++)
        {
            attr_inv.put(attributes.get(i),i);
        }
    }
    return attr_inv.get(attr);
  }

  /**
   * Helper function to get the index of the attributeValue in the list for the attribute key in the attributeValues map
   */
  private int getAttributeValueIndex(String attr, String value) {
    for (int i = 0; i < attributeValues.get(attr).size(); i++) {
      if (value.equals(attributeValues.get(attr).get(i))) {
        return i;
      }
    }
    return -1;
  }
}
