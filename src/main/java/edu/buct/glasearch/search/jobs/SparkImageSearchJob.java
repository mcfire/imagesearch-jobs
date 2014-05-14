package edu.buct.glasearch.search.jobs;

import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.imageio.ImageIO;

import net.semanticmetadata.lire.imageanalysis.EdgeHistogram;
import net.semanticmetadata.lire.imageanalysis.LireFeature;
import net.semanticmetadata.lire.imageanalysis.SimpleColorHistogram;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.CompatibilityFactory;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.client.Get;
import org.apache.hadoop.hbase.client.HTable;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.JobUtil;
import org.apache.hadoop.hbase.mapreduce.TableInputFormat;
import org.apache.hadoop.hbase.protobuf.generated.HBaseProtos;
import org.apache.hadoop.hbase.thrift2.ThriftServer;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.hbase.util.vint.UVLongTool;
import org.apache.hadoop.mapred.JobConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;

import com.google.protobuf.HBaseZeroCopyByteString;
import com.google.protobuf.LazyStringArrayList;

import scala.Tuple2;
import edu.buct.glasearch.search.jobs.ImageSearchJob.FeatureObject;

public class SparkImageSearchJob implements Serializable {

	private static final long serialVersionUID = 8533489548835413763L;
	private static final Log logger = LogFactory.getLog(SparkImageSearchJob.class);

	public static class Map extends PairFlatMapFunction<Tuple2<ImmutableBytesWritable, Result>, 
												Double, FeatureObject> {

		private static final long serialVersionUID = 8423288760249384303L;
		
		//检索信息的rowId，检索结果存储于HBase中
		String searchRowId = null;
		//待检索图像的颜色直方图信息
		LireFeature colorFeature = null;
		//待检索图像的边缘直方图信息
		LireFeature edgeFeature = null;
		//JSON格式转换工具
		
		double colorAvg, colorSigma, edgeAvg, edgeSigma;

		@Override
		public Iterable<Tuple2<Double, FeatureObject>> call(
				Tuple2<ImmutableBytesWritable, Result> input) throws Exception {

			ImmutableBytesWritable rowKey = input._1();
			Result result = input._2();
			
			//图像信息表中当前行的rowId
			String rowId = new String(rowKey.get());
			
			if (logger.isInfoEnabled()) {
				logger.info("map operation on:" + rowId);
			}
			
			//读取目标图像的两类直方图特征
			byte[] targetColorFeatureBytes = result.getValue(
					ImageSearchJob.COLUMN_FAMILY_BYTES, ImageSearchJob.COLOR_FEATURE_COLUMN);
			byte[] targetEdgeFeatureBytes = result.getValue(
					ImageSearchJob.COLUMN_FAMILY_BYTES, ImageSearchJob.EDGE_FEATURE_COLUMN);
			if (targetColorFeatureBytes == null || targetEdgeFeatureBytes == null) {
				logger.error("some feature is null");
				return new ArrayList();
			}
			
			//将两类直方图特征以对象的形式表示，并计算和待检索图像特征之间的距离
			LireFeature targetColorFeature = new SimpleColorHistogram();
			targetColorFeature.setByteArrayRepresentation(targetColorFeatureBytes);
			float colorDistance = targetColorFeature.getDistance(colorFeature);//计算和待检索图像特征之间的距离
			double flatColorDistance = (colorDistance - colorAvg) / colorSigma;//距离归一化
			FeatureObject colorFeatureObject = new FeatureObject(
					rowId, (float)flatColorDistance, FeatureObject.FeatureType.color);//生成结果对象
			Tuple2<Double,ImageSearchJob.FeatureObject> colorTuple = new Tuple2(flatColorDistance, colorFeatureObject);
			
			LireFeature targetEdgeFeature = new EdgeHistogram();
			targetEdgeFeature.setByteArrayRepresentation(targetEdgeFeatureBytes);
			float edgeDistance = targetEdgeFeature.getDistance(edgeFeature);//计算和待检索图像特征之间的距离
			double flatEdgeDistance = (edgeDistance - edgeAvg) / edgeSigma;	//距离归一化
			FeatureObject edgeFeatureObject = new FeatureObject(
					rowId, (float)flatEdgeDistance, FeatureObject.FeatureType.edge);//生成距离对象
			Tuple2<Double,ImageSearchJob.FeatureObject> edgeTuple = new Tuple2(flatEdgeDistance, edgeFeatureObject);
			
			//以JSON的格式将特征距离对象写入到Map的输入。输出按照特征类型分类
			return Arrays.asList(colorTuple, edgeTuple);
		}
	
		public void setup(LireFeature colorFeature, LireFeature edgeFeature, 
				double colorAvg, double colorSigma, double edgeAvg, double edgeSigma, Configuration config) throws IOException,
				InterruptedException {
			
			this.colorFeature = colorFeature;
			this.edgeFeature = edgeFeature;
			this.colorAvg = colorAvg;
			this.colorSigma = colorSigma;
			this.edgeAvg = edgeAvg;
			this.edgeSigma = edgeSigma;
			
		}
	}
	
	private static void addToList(String[] array, Set<String> list) {
		if (array == null || list == null) return;
		for (String element : array) {
			list.add(element);
		}
	}

	public static void main(String[] args) throws IOException, InterruptedException {
		
		//String masterAddress = "local";
		String masterAddress = "spark://cluster1.centos:7077";
		
		Set<String> jars = new HashSet<String>();

		addToList(JavaSparkContext.jarOfClass(SparkImageSearchJob.class), jars);
		addToList(JavaSparkContext.jarOfClass(SimpleColorHistogram.class), jars);

		addToList(JavaSparkContext.jarOfClass(StringUtils.class), jars);
		addToList(JavaSparkContext.jarOfClass(LogFactory.class), jars);
		addToList(JavaSparkContext.jarOfClass(Configuration.class), jars);
		addToList(JavaSparkContext.jarOfClass(HTable.class), jars);
		addToList(JavaSparkContext.jarOfClass(TableInputFormat.class), jars);
		addToList(JavaSparkContext.jarOfClass(HBaseConfiguration.class), jars);
		addToList(JavaSparkContext.jarOfClass(HBaseProtos.class), jars);
		addToList(JavaSparkContext.jarOfClass(JobUtil.class), jars);
		addToList(JavaSparkContext.jarOfClass(CompatibilityFactory.class), jars);
		addToList(JavaSparkContext.jarOfClass(UVLongTool.class), jars);
		addToList(JavaSparkContext.jarOfClass(ThriftServer.class), jars);
		addToList(JavaSparkContext.jarOfClass(org.cloudera.htrace.Trace.class), jars);
		
		addToList(JavaSparkContext.jarOfClass(HBaseZeroCopyByteString.class), jars);
		addToList(JavaSparkContext.jarOfClass(LazyStringArrayList.class), jars);

//		String classPath = "";
//		File libs = new File(classPath);
//		for (String lib : libs.list()) {
//			if (!lib.endsWith(".jar")) continue;
//			jars.add(lib);
//		}
		
		JavaSparkContext ctx = new JavaSparkContext(
				masterAddress, 
				"SparkJob",
				System.getenv("SPARK_HOME"),
				jars.toArray(new String[jars.size()]));
		
		JobConf jobConf = new JobConf();
		jobConf.set("fs.defaultFS", "hdfs://cluster1.centos:8020");
		jobConf.set("yarn.resourcemanager.address", "cluster1.centos:8032");
		jobConf.set("mapreduce.framework.name", "yarn");
		
	    String imageFileName = "/root/development/data/images/39/39115.jpg";
		
		Scan scan = new Scan();
		scan.setCaching(500);        // 1 is the default in Scan, which will be bad for MapReduce jobs
		scan.setCacheBlocks(false);  // don't set to true for MR jobs
		
		jobConf.set(TableInputFormat.INPUT_TABLE, ImageSearchJob.imageInfoTable);
		// read data
		JavaPairRDD<ImmutableBytesWritable, Result> hbaseData = ctx.newAPIHadoopRDD(jobConf, 
				TableInputFormat.class, 
				ImmutableBytesWritable.class, Result.class);
		
	    
	    BufferedImage image = ImageIO.read(new FileInputStream(imageFileName));
		LireFeature colorFeature = new SimpleColorHistogram();
		colorFeature.extract(image);
		LireFeature edgeFeature = new EdgeHistogram();
		edgeFeature.extract(image);

		
		//从距离信息表中提取图像间距离的平均值和方差，用于距离的归一化。
		HTable distanceTable = new HTable(jobConf, ImageSearchJob.imageDistanceTable);
		Get colorDistanceGet = new Get(ImageSearchJob.COLOR_FEATURE_RESULT_COLUMN);
		Result colorDistance = distanceTable.get(colorDistanceGet);
		double colorAvg = Bytes.toDouble(colorDistance.getValue(
				ImageSearchJob.COLUMN_FAMILY_BYTES, Bytes.toBytes("avg")));
		double colorSigma = Bytes.toDouble(colorDistance.getValue(
				ImageSearchJob.COLUMN_FAMILY_BYTES, Bytes.toBytes("sigma")));
		
		Get edgeDistanceGet = new Get(ImageSearchJob.EDGE_FEATURE_RESULT_COLUMN);
		Result edgeDistance = distanceTable.get(edgeDistanceGet);
		double edgeAvg = Bytes.toDouble(edgeDistance.getValue(
				ImageSearchJob.COLUMN_FAMILY_BYTES, Bytes.toBytes("avg")));
		double edgeSigma = Bytes.toDouble(edgeDistance.getValue(
				ImageSearchJob.COLUMN_FAMILY_BYTES, Bytes.toBytes("sigma")));
		
		distanceTable.close();
		
		Map map = new Map();
		map.setup(colorFeature, edgeFeature, colorAvg, colorSigma, edgeAvg, edgeSigma, jobConf);
		
		hbaseData.flatMap(map).collect();
		
		JavaPairRDD<Double, FeatureObject> colorRdd = hbaseData.flatMap(map);
		JavaPairRDD<Double, FeatureObject> edgeRdd = colorRdd.cache();
		
		List<Tuple2<Double, FeatureObject>> colorResult = colorRdd.filter(new Function<Tuple2<Double,ImageSearchJob.FeatureObject>,Boolean>() {

			private static final long serialVersionUID = -8897212007359320125L;

			@Override
			public Boolean call(Tuple2<Double, FeatureObject> feature)
					throws Exception {
				return feature._2.getType() == FeatureObject.FeatureType.color;
			}
		}).sortByKey(false).collect();
		
		List<Tuple2<Double, FeatureObject>> edgeResult = edgeRdd.filter(new Function<Tuple2<Double,ImageSearchJob.FeatureObject>,Boolean>() {

			private static final long serialVersionUID = -4533280931765784913L;

			@Override
			public Boolean call(Tuple2<Double, FeatureObject> feature)
					throws Exception {
				return feature._2.getType() == FeatureObject.FeatureType.edge;
			}
		}).sortByKey(false).collect();

		edgeResult.size();
	}

}
