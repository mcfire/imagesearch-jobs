package edu.buct.glasearch.search.jobs;

import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.Serializable;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.UUID;

import javax.imageio.ImageIO;

import net.semanticmetadata.lire.imageanalysis.EdgeHistogram;
import net.semanticmetadata.lire.imageanalysis.LireFeature;
import net.semanticmetadata.lire.imageanalysis.SimpleColorHistogram;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.hbase.client.Get;
import org.apache.hadoop.hbase.client.HTable;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.TableMapReduceUtil;
import org.apache.hadoop.hbase.mapreduce.TableMapper;
import org.apache.hadoop.hbase.mapreduce.TableReducer;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;

import com.google.gson.Gson;

public class ImageSearchJob {

	private static final Log logger = LogFactory.getLog(ImageSearchJob.class);
	
	//图像标题的列名
	public static final byte[] TITLE_COLUMN_BYTES = "title".getBytes();
	
	public static final byte[] FILENAME_COLUMN_BYTES = "image".getBytes();
	
	public static final byte[] LAT_COLUMN_BYTES = "lat".getBytes();

	public static final byte[] LNG_COLUMN_BYTES = "lng".getBytes();

	//颜色直方图特征的列名
	public static final String COLOR_FEATURE = "color_f";
	//边缘直方图特征的列名
	public static final String EDGE_FEATURE = "edge_f";
	public static final byte[] COLOR_FEATURE_COLUMN = Bytes.toBytes(COLOR_FEATURE);
	public static final byte[] EDGE_FEATURE_COLUMN = Bytes.toBytes(EDGE_FEATURE);
	
	//颜色直方图的匹配结果Key
	public static final String COLOR_FEATURE_RESULT = "color_r";
	//边缘直方图的匹配结果Key
	public static final String EDGE_FEATURE_RESULT = "edge_r";
	public static final byte[] COLOR_FEATURE_RESULT_COLUMN = Bytes.toBytes(COLOR_FEATURE_RESULT);
	public static final byte[] EDGE_FEATURE_RESULT_COLUMN = Bytes.toBytes(EDGE_FEATURE_RESULT);
	
	//检索信息的rowId，检索结果存储于HBase中
	public static final String SEARCH_ROWID = "search_rowid";

	//HBase列族名
	public static final String COLUMN_FAMILY = "i";

	public static final byte[] COLUMN_FAMILY_BYTES = Bytes.toBytes(COLUMN_FAMILY);
	
	//图像信息表名
	public static final String imageInfoTable = "imageinfo";
	
	public static ImmutableBytesWritable imageInfoTableBytes = 
			new ImmutableBytesWritable(Bytes.toBytes(imageInfoTable));
	
	//检索配置和结果表名
	public static final String imageResultTable = "imageresult";
	
	public static ImmutableBytesWritable imageResultTableBytes = 
			new ImmutableBytesWritable(Bytes.toBytes(imageResultTable));
	
	//检索配置和结果表名
	public static final String imageDistanceTable = "imagedistance";

	public static class Map extends TableMapper<Text, Text> {
		//文件系统对象，用于访问HDFS上的图像文件
		FileSystem fs = null;
		//检索信息的rowId，检索结果存储于HBase中
		String searchRowId = null;
		//待检索图像的颜色直方图信息
		SimpleColorHistogram colorFeature = null;
		//待检索图像的边缘直方图信息
		EdgeHistogram edgeFeature = null;
		//JSON格式转换工具
		Gson gson = new Gson();
		
		double colorAvg, colorSigma, edgeAvg, edgeSigma;
	
		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			
			Configuration config = context.getConfiguration();
			//初始化文件系统对象，连接到Hadoop平台的HDFS上
			fs = FileSystem.get(config);
			
			//接收任务调用者传递的检索信息表rowId
			searchRowId = config.get(SEARCH_ROWID);
			HTable table = new HTable(config, imageResultTable);
			Get featuresGet = new Get(searchRowId.getBytes());
			Result result = table.get(featuresGet);	//根据searchRowId读取检索信息
			
			//读取待检索图像的颜色直方图特征和边缘直方图特征
			byte[] colorFeatureBytes = result.getValue(COLUMN_FAMILY_BYTES, COLOR_FEATURE_COLUMN);
			byte[] edgeFeatureBytes = result.getValue(COLUMN_FAMILY_BYTES, EDGE_FEATURE_COLUMN);
			
			//将两类特征封装为对象形式
			colorFeature = new SimpleColorHistogram();
			colorFeature.setByteArrayRepresentation(colorFeatureBytes);
			
			edgeFeature = new EdgeHistogram();
			edgeFeature.setByteArrayRepresentation(edgeFeatureBytes);
			
			table.close();
			
			//从距离信息表中提取图像间距离的平均值和方差，用于距离的归一化。
			HTable distanceTable = new HTable(config, imageDistanceTable);
			Get colorDistanceGet = new Get(COLOR_FEATURE_RESULT_COLUMN);
			Result colorDistance = distanceTable.get(colorDistanceGet);
			colorAvg = Bytes.toDouble(colorDistance.getValue(COLUMN_FAMILY_BYTES, Bytes.toBytes("avg")));
			colorSigma = Bytes.toDouble(colorDistance.getValue(COLUMN_FAMILY_BYTES, Bytes.toBytes("sigma")));
			
			Get edgeDistanceGet = new Get(EDGE_FEATURE_RESULT_COLUMN);
			Result edgeDistance = distanceTable.get(edgeDistanceGet);
			edgeAvg = Bytes.toDouble(edgeDistance.getValue(COLUMN_FAMILY_BYTES, Bytes.toBytes("avg")));
			edgeSigma = Bytes.toDouble(edgeDistance.getValue(COLUMN_FAMILY_BYTES, Bytes.toBytes("sigma")));
			
			distanceTable.close();
		}
		
		@Override
		protected void map(ImmutableBytesWritable rowKey, Result result,
				Context context) throws IOException, InterruptedException {
			//图像信息表中当前行的rowId
			String rowId = new String(rowKey.get());
			
			if (logger.isInfoEnabled()) {
				logger.info("map operation on:" + rowId);
			}
			
			//读取目标图像的两类直方图特征
			byte[] targetColorFeatureBytes = result.getValue(COLUMN_FAMILY_BYTES, COLOR_FEATURE_COLUMN);
			byte[] targetEdgeFeatureBytes = result.getValue(COLUMN_FAMILY_BYTES, EDGE_FEATURE_COLUMN);
			if (targetColorFeatureBytes == null || targetEdgeFeatureBytes == null) {
				logger.error("some feature is null");
				return;
			}
			
			//将两类直方图特征以对象的形式表示，并计算和待检索图像特征之间的距离
			LireFeature targetColorFeature = new SimpleColorHistogram();
			targetColorFeature.setByteArrayRepresentation(targetColorFeatureBytes);
			float colorDistance = targetColorFeature.getDistance(colorFeature);//计算和待检索图像特征之间的距离
			double flatColorDistance = (colorDistance - colorAvg) / colorSigma;//距离归一化
			FeatureObject colorFeatureObject = new FeatureObject(
					rowId, (float)flatColorDistance, FeatureObject.FeatureType.color);//生成结果对象
			
			LireFeature targetEdgeFeature = new EdgeHistogram();
			targetEdgeFeature.setByteArrayRepresentation(targetEdgeFeatureBytes);
			float edgeDistance = targetEdgeFeature.getDistance(edgeFeature);//计算和待检索图像特征之间的距离
			double flatEdgeDistance = (edgeDistance - edgeAvg) / edgeSigma;	//距离归一化
			FeatureObject edgeFeatureObject = new FeatureObject(
					rowId, (float)flatEdgeDistance, FeatureObject.FeatureType.color);//生成距离对象
			
			//以JSON的格式将特征距离对象写入到Map的输入。输出按照特征类型分类
			context.write(new Text(COLOR_FEATURE_RESULT), new Text(gson.toJson(colorFeatureObject)));
			context.write(new Text(EDGE_FEATURE_RESULT), new Text(gson.toJson(edgeFeatureObject)));
		}
	}

	public static class Reduce extends TableReducer<Text, Text, ImmutableBytesWritable> {
		//默认检索结果数量
		int resultSize = 10;
		//检索信息的rowId，检索结果存储于HBase中
		String searchRowId = null;
		//JSON格式转换工具
		Gson gson = new Gson();
	
		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			//读取任务调用方传递的检索参数
			Configuration config = context.getConfiguration();
			searchRowId = config.get(SEARCH_ROWID);
			resultSize = config.getInt("resultSize", resultSize);
		}
		
		@Override
		protected void reduce(Text rowKey, Iterable<Text> values,
				Context context) throws IOException, InterruptedException {
			
			List<FeatureObject> features = new LinkedList<FeatureObject>();
			//将距离对象进行JSON格式转换
			Iterator<Text> itor = values.iterator();
			while (itor.hasNext()) {
				String json = itor.next().toString();
				FeatureObject feature = gson.fromJson(json, FeatureObject.class);
				features.add(feature);				
			}
			
			//对结果进行按照距离排序
			Collections.sort(features);
			//取指定数量的结果
			if (resultSize < features.size()) {
				features = features.subList(0, resultSize);
			}
			//将结果转换为JSON格式
			FeatureList resultObject = new FeatureList(features);
			String result = gson.toJson(resultObject);
			
			//将结果写入输出，Reduce操作完成后将写入HBase数据库
			Put put = new Put(searchRowId.getBytes());
			put.add(COLUMN_FAMILY_BYTES, rowKey.toString().getBytes(), result.getBytes());
			context.write(null, put);
		}
	}
	
	public static class FeatureObject implements Comparable<FeatureObject>, Serializable {
		
		public enum FeatureType {
			color,
			edge
		}
		
		private String rowId;

		private float distance;
		
		private FeatureType type;

		public FeatureType getType() {
			return type;
		}

		public void setType(FeatureType type) {
			this.type = type;
		}

		public FeatureObject(String rowId, float distance, FeatureType type) {
			super();
			this.rowId = rowId;
			this.distance = distance;
			this.type = type;
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
			
			if (this.type.ordinal() != o.type.ordinal()) {
				return new Integer(this.type.ordinal()).compareTo(o.type.ordinal());
			}
			return this.distance > o.distance ? 1 : (this.distance == o.distance ? 0 : -1);
		}
		
		@Override
		public boolean equals(Object o) {
			if (this.rowId == null || o == null || !(o instanceof FeatureObject)) return false;
			return this.rowId.equals(((FeatureObject)o).rowId);
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
	public static Job configureJob(Configuration conf, byte[] startRow, byte[] stopRow)
			throws IOException {

		//important: use this method to add job and it's dependency jar
		TableMapReduceUtil.addDependencyJars(conf, ImageSearchJob.class, LireFeature.class, Gson.class);
		
		JobConf jobConf = new JobConf(conf);
		jobConf.setJobName("image-search");
		
		Job job = new Job(jobConf);
		job.setJarByClass(ImageIndexJob.class);
		
		Scan scan = new Scan();
		scan.setCaching(500);        // 1 is the default in Scan, which will be bad for MapReduce jobs
		scan.setCacheBlocks(false);  // don't set to true for MR jobs
		scan.setStartRow(startRow);
		scan.setStopRow(stopRow);
		// set other scan attrs
		
		TableMapReduceUtil.initTableMapperJob(
				imageInfoTable,        // input table
				scan,               // Scan instance to control CF and attribute selection
				Map.class,     // mapper class
				Text.class,         // mapper output key
				Text.class,  // mapper output value
				job);
		
		TableMapReduceUtil.initTableReducerJob(
				imageResultTable,      // output table
				Reduce.class,             // reducer class
				job);
		job.setNumReduceTasks(1);
		
		return job;
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
	    //conf.set("fs.defaultFS", "hdfs://cluster1.centos:8020");
	    //conf.set("yarn.resourcemanager.address", "cluster1.centos:8032");
	    //conf.set("mapreduce.framework.name", "yarn");
	    String imageFileName = "/root/development/data/images/39/39115.jpg";
		String START_ROW = "39";
		String STOP_ROW = "39-393300";
	    
	    BufferedImage image = ImageIO.read(new FileInputStream(imageFileName));
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
	    
		Job job = configureJob(conf, Bytes.toBytes(START_ROW),  Bytes.toBytes(STOP_ROW));

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
