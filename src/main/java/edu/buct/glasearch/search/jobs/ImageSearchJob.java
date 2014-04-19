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

public class ImageSearchJob {

	public static final byte[] TITLE_COLUMN_BYTES = "title".getBytes();

	public static final byte[] LAT_COLUMN_BYTES = "lat".getBytes();

	public static final byte[] LNG_COLUMN_BYTES = "lng".getBytes();

	public static final String COLOR_FEATURE = "color_f";
	public static final String EDGE_FEATURE = "edge_f";
	public static final byte[] COLOR_FEATURE_COLUMN = Bytes.toBytes(COLOR_FEATURE);
	public static final byte[] EDGE_FEATURE_COLUMN = Bytes.toBytes(EDGE_FEATURE);
	
	public static final String COLOR_FEATURE_RESULT = "color_r";
	public static final String EDGE_FEATURE_RESULT = "edge_r";
	public static final byte[] COLOR_FEATURE_RESULT_COLUMN = Bytes.toBytes(COLOR_FEATURE_RESULT);
	public static final byte[] EDGE_FEATURE_RESULT_COLUMN = Bytes.toBytes(EDGE_FEATURE_RESULT);
	
	public static final String SEARCH_ROWID = "search_rowid";

	public static final String COLUMN_FAMILY = "i";

	public static final byte[] COLUMN_FAMILY_BYTES = Bytes.toBytes(COLUMN_FAMILY);
	
	public static final String imageInfoTable = "imageinfo";
	
	public static ImmutableBytesWritable imageInfoTableBytes = 
			new ImmutableBytesWritable(Bytes.toBytes(imageInfoTable));
	
	public static final String imageResultTable = "imageresult";
	
	public static ImmutableBytesWritable imageResultTableBytes = 
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
			
			byte[] targetColorFeatureBytes = result.getValue(COLUMN_FAMILY_BYTES, COLOR_FEATURE_COLUMN);
			byte[] targetEdgeFeatureBytes = result.getValue(COLUMN_FAMILY_BYTES, EDGE_FEATURE_COLUMN);
			
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
			
			byte[] colorFeatureBytes = result.getValue(COLUMN_FAMILY_BYTES, COLOR_FEATURE_COLUMN);
			byte[] edgeFeatureBytes = result.getValue(COLUMN_FAMILY_BYTES, EDGE_FEATURE_COLUMN);
			
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
			
			FeatureList resultObject = new FeatureList(features);
			String result = gson.toJson(resultObject);

			Put put = new Put(searchRowId.getBytes());
			put.add(COLUMN_FAMILY_BYTES, rowKey.toString().getBytes(), result.getBytes());
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
	
	public static class FeatureList {
		
		private List<FeatureObject> result;
		
		public FeatureList() {}

		public FeatureList(List<FeatureObject> result) {
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
		job.setJarByClass(ImageSearchJob.class);
		job.setMapperClass(ImageSearchJob.Map.class);
		job.setReducerClass(ImageSearchJob.Reduce.class);
		job.setInputFormatClass(TableInputFormat.class);
		job.setMapOutputKeyClass(Text.class);
		job.setOutputFormatClass(MultiTableOutputFormat.class);
		return job;
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		
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
	    featuresPut.add(COLUMN_FAMILY_BYTES, COLOR_FEATURE_COLUMN, colorFeature.getByteArrayRepresentation());
	    featuresPut.add(COLUMN_FAMILY_BYTES, EDGE_FEATURE_COLUMN, edgeFeature.getByteArrayRepresentation());
	    table.put(featuresPut);
	    
	    conf.set(SEARCH_ROWID, rowId);
	    
		Job job = configureJob(conf);

		boolean isSuccess = job.waitForCompletion(true);
		
		if (isSuccess) {
			Gson gson = new Gson();
			
			Get featuresGet = new Get(rowId.getBytes());
			Result result = table.get(featuresGet);
			byte[] colorFeatureResultBytes = result.getValue(COLUMN_FAMILY_BYTES, COLOR_FEATURE_RESULT_COLUMN);
			String colorFeatureResultJson = new String(colorFeatureResultBytes);
			FeatureList colorFeatureResult = gson.fromJson(colorFeatureResultJson, FeatureList.class);

			byte[] edgeFeatureResultBytes = result.getValue(COLUMN_FAMILY_BYTES, EDGE_FEATURE_RESULT_COLUMN);
			String edgeFeatureResultJson = new String(edgeFeatureResultBytes);
			FeatureList edgeFeatureResult = gson.fromJson(edgeFeatureResultJson, FeatureList.class);
			//TODO we got the result
		}
		
	    table.close();
	    
		System.exit(isSuccess ? 0 : -1);
	}
}
