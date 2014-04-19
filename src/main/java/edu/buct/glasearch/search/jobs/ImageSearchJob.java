/**
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.buct.glasearch.search.jobs;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.UUID;

import net.semanticmetadata.lire.imageanalysis.EdgeHistogram;
import net.semanticmetadata.lire.imageanalysis.LireFeature;
import net.semanticmetadata.lire.imageanalysis.SimpleColorHistogram;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.client.Get;
import org.apache.hadoop.hbase.client.HTable;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.MultiTableOutputFormat;
import org.apache.hadoop.hbase.mapreduce.TableInputFormat;
import org.apache.hadoop.hbase.protobuf.ProtobufUtil;
import org.apache.hadoop.hbase.protobuf.generated.ClientProtos;
import org.apache.hadoop.hbase.util.Base64;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import com.google.gson.Gson;

/**
 * Example map/reduce job to construct index tables that can be used to quickly
 * find a row based on the value of a column. It demonstrates:
 * <ul>
 * <li>Using TableInputFormat and TableMapReduceUtil to use an HTable as input
 * to a map/reduce job.</li>
 * <li>Passing values from main method to children via the configuration.</li>
 * <li>Using MultiTableOutputFormat to output to multiple tables from a
 * map/reduce job.</li>
 * <li>A real use case of building a secondary index over a table.</li>
 * </ul>
 * 
 * <h3>Usage</h3>
 * 
 * <p>
 * Modify ${HADOOP_HOME}/conf/hadoop-env.sh to include the hbase jar, the
 * zookeeper jar (can be found in lib/ directory under HBase root, the examples
 * output directory, and the hbase conf directory in HADOOP_CLASSPATH, and then
 * run
 * <tt><strong>bin/hadoop org.apache.hadoop.hbase.mapreduce.IndexBuilder TABLE_NAME COLUMN_FAMILY ATTR [ATTR ...]</strong></tt>
 * </p>
 * 
 * <p>
 * To run with the sample data provided in index-builder-setup.rb, use the
 * arguments <strong><tt>people attributes name email phone</tt></strong>.
 * </p>
 * 
 * <p>
 * This code was written against HBase 0.21 trunk.
 * </p>
 */
public class ImageSearcher {

	public static final String COLOR_FEATURE = "color_f";
	public static final String EDGE_FEATURE = "edge_f";
	public static final byte[] COLOR_FEATURE_COLUMN = Bytes.toBytes(COLOR_FEATURE);
	public static final byte[] EDGE_FEATURE_COLUMN = Bytes.toBytes(EDGE_FEATURE);
	
	public static final String COLOR_FEATURE_RESULT = "color_r";
	public static final String EDGE_FEATURE_RESULT = "edge_r";
	public static final byte[] COLOR_FEATURE_RESULT_COLUMN = Bytes.toBytes(COLOR_FEATURE_RESULT);
	public static final byte[] EDGE_FEATURE_RESULT_COLUMN = Bytes.toBytes(EDGE_FEATURE_RESULT);
	
	public static final String SEARCH_ROWID = "search_rowid";

	public static final byte[] COLUMN_FAMILY = Bytes.toBytes("i");
	
	private static final String imageInfoTable = "imageinfo";
	
	private static ImmutableBytesWritable imageInfoTableBytes = 
			new ImmutableBytesWritable(Bytes.toBytes(imageInfoTable));
	
	private static final String imageResultTable = "imageresult";
	
	private static ImmutableBytesWritable imageResultTableBytes = 
			new ImmutableBytesWritable(Bytes.toBytes(imageResultTable));

	public static class Map extends Mapper<ImmutableBytesWritable, Result, Text, Text> {
		
		FileSystem fs = null;
		
		String searchRowId = null;
		
		SimpleColorHistogram colorFeature = null;
		
		EdgeHistogram edgeFeature = null;
		
		Gson gson = new Gson();
		
		@Override
		protected void map(ImmutableBytesWritable rowKey, Result result,
				Context context) throws IOException, InterruptedException {
			
			String rowId = new String(rowKey.get());
			
			byte[] targetColorFeatureBytes = result.getValue(COLUMN_FAMILY, COLOR_FEATURE_COLUMN);
			byte[] targetEdgeFeatureBytes = result.getValue(COLUMN_FAMILY, EDGE_FEATURE_COLUMN);
			
			LireFeature targetColorFeature = new SimpleColorHistogram();
			targetColorFeature.setByteArrayRepresentation(targetColorFeatureBytes);
			float colorDistance = targetColorFeature.getDistance(colorFeature);
			FeatureObject colorFeatureObject = new FeatureObject(rowId, colorDistance);
			
			LireFeature targetEdgeFeature = new EdgeHistogram();
			targetEdgeFeature.setByteArrayRepresentation(targetEdgeFeatureBytes);
			float edgeDistance = targetEdgeFeature.getDistance(edgeFeature);
			FeatureObject edgeFeatureObject = new FeatureObject(rowId, edgeDistance);
			
			context.write(new Text(COLOR_FEATURE_RESULT), new Text(gson.toJson(colorFeatureObject)));
			context.write(new Text(EDGE_FEATURE_RESULT), new Text(gson.toJson(edgeFeatureObject)));
		}
	
		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			
			Configuration config = context.getConfiguration();
			
			fs = FileSystem.get(config);
			
			searchRowId = config.get(SEARCH_ROWID);
			HTable table = new HTable(config, imageResultTable);
			Get featuresGet = new Get(searchRowId.getBytes());
			Result result = table.get(featuresGet);
			
			byte[] colorFeatureBytes = result.getValue(COLUMN_FAMILY, COLOR_FEATURE_COLUMN);
			byte[] edgeFeatureBytes = result.getValue(COLUMN_FAMILY, EDGE_FEATURE_COLUMN);
			
			colorFeature = new SimpleColorHistogram();
			colorFeature.setByteArrayRepresentation(colorFeatureBytes);
			
			edgeFeature = new EdgeHistogram();
			edgeFeature.setByteArrayRepresentation(edgeFeatureBytes);
			
			table.close();
		}
	}

	public static class Reduce extends Reducer<Text, Text, ImmutableBytesWritable, Put> {
		
		int defaultResultSize = 10;
		
		String searchRowId = null;
		
		Gson gson = new Gson();
		
		@Override
		protected void reduce(Text rowKey, Iterable<Text> values,
				Context context) throws IOException, InterruptedException {
			
			List<FeatureObject> features = new LinkedList<FeatureObject>();
			
			Iterator<Text> itor = values.iterator();
			while (itor.hasNext()) {
				String json = itor.next().toString();
				FeatureObject feature = gson.fromJson(json, FeatureObject.class);
				features.add(feature);				
			}
			
			Collections.sort(features);
			features = features.subList(0, defaultResultSize);
			
			SearchResult resultObject = new SearchResult(features);
			String result = gson.toJson(resultObject);

			Put put = new Put(searchRowId.getBytes());
			put.add(COLUMN_FAMILY, rowKey.toString().getBytes(), result.getBytes());
			context.write(imageResultTableBytes, put);
		}
	
		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			
			Configuration config = context.getConfiguration();
			searchRowId = config.get(SEARCH_ROWID);
			defaultResultSize = config.getInt("defaultResultSize", defaultResultSize);
		}
	}
	
	public static class FeatureObject implements Comparable<FeatureObject> {
		
		private String rowId;

		private float distance;

		public FeatureObject(String rowId, float distance) {
			super();
			this.rowId = rowId;
			this.distance = distance;
		}

		public String getRowId() {
			return rowId;
		}

		public void setRowId(String rowId) {
			this.rowId = rowId;
		}

		public float getDistance() {
			return distance;
		}

		public void setDistance(float distance) {
			this.distance = distance;
		}

		@Override
		public int compareTo(FeatureObject o) {
			if (o == null) return 1;
			
			return this.distance > o.distance ? 1 : (this.distance == o.distance ? 0 : -1);
		}
	}
	
	public static class SearchResult {
		
		private List<FeatureObject> result;

		public SearchResult(List<FeatureObject> result) {
			super();
			this.result = result;
		}

		public List<FeatureObject> getResult() {
			return result;
		}

		public void setResult(List<FeatureObject> result) {
			this.result = result;
		}
		
	}

	/**
	 * Job configuration.
	 */
	public static Job configureJob(Configuration conf)
			throws IOException {

		ClientProtos.Scan proto = ProtobufUtil.toScan(new Scan());
		String scanString = Base64.encodeBytes(proto.toByteArray());

		JobConf jobConf = new JobConf(conf);
		jobConf.setJobName("image-search");
		
		jobConf.set(TableInputFormat.SCAN, scanString);
		jobConf.set(TableInputFormat.INPUT_TABLE, imageInfoTable);
		Job job = new Job(jobConf);
		job.setJarByClass(ImageSearcher.class);
		job.setMapperClass(ImageSearcher.Map.class);
		job.setReducerClass(ImageSearcher.Reduce.class);
		job.setInputFormatClass(TableInputFormat.class);
		job.setMapOutputKeyClass(Text.class);
		job.setOutputFormatClass(MultiTableOutputFormat.class);
		return job;
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = HBaseConfiguration.create();
		
	    conf.set("fs.defaultFS", "hdfs://cluster1.centos:8020");
	    //conf.set("yarn.resourcemanager.address", "cluster1.centos:8032");
	    //conf.set("mapreduce.framework.name", "yarn");
	    
	    
	    BufferedImage image = new BufferedImage(300, 600, BufferedImage.TYPE_INT_RGB);
		LireFeature colorFeature = new SimpleColorHistogram();
		colorFeature.extract(image);
		LireFeature edgeFeature = new EdgeHistogram();
		edgeFeature.extract(image);

		String rowId = UUID.randomUUID().toString();

	    HTable table = new HTable(conf, imageResultTable);
	    Put featuresPut = new Put(rowId.getBytes());
	    featuresPut.add(COLUMN_FAMILY, COLOR_FEATURE_COLUMN, colorFeature.getByteArrayRepresentation());
	    featuresPut.add(COLUMN_FAMILY, EDGE_FEATURE_COLUMN, edgeFeature.getByteArrayRepresentation());
	    table.put(featuresPut);
	    
	    conf.set(SEARCH_ROWID, rowId);
	    
		Job job = configureJob(conf);

		boolean isSuccess = job.waitForCompletion(true);
		
		if (isSuccess) {
			Gson gson = new Gson();
			
			Get featuresGet = new Get(rowId.getBytes());
			Result result = table.get(featuresGet);
			byte[] colorFeatureResultBytes = result.getValue(COLUMN_FAMILY, COLOR_FEATURE_RESULT_COLUMN);
			String colorFeatureResultJson = new String(colorFeatureResultBytes);
			SearchResult colorFeatureResult = gson.fromJson(colorFeatureResultJson, SearchResult.class);

			byte[] edgeFeatureResultBytes = result.getValue(COLUMN_FAMILY, EDGE_FEATURE_RESULT_COLUMN);
			String edgeFeatureResultJson = new String(edgeFeatureResultBytes);
			SearchResult edgeFeatureResult = gson.fromJson(edgeFeatureResultJson, SearchResult.class);
			//TODO we got the result
		}
		
	    table.close();
	    
		System.exit(isSuccess ? 0 : -1);
	}
}
