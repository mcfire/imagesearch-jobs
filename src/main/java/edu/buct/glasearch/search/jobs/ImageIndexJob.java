package edu.buct.glasearch.search.jobs;

import java.awt.image.BufferedImage;
import java.io.IOException;

import javax.imageio.ImageIO;

import net.semanticmetadata.lire.imageanalysis.EdgeHistogram;
import net.semanticmetadata.lire.imageanalysis.LireFeature;
import net.semanticmetadata.lire.imageanalysis.SimpleColorHistogram;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.TableMapReduceUtil;
import org.apache.hadoop.hbase.mapreduce.TableMapper;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;

public class ImageIndexJob extends 
	TableMapper<ImmutableBytesWritable, Put> {
	//hbase的列族
	public static final byte[] COLUMN_FAMILY = Bytes.toBytes("i");
	//颜色直方图特征的列名
	public static final byte[] COLOR_FEATURE_COLUMN = Bytes.toBytes("color_f");
	//边缘直方图特征的列名
	public static final byte[] EDGE_FEATURE_COLUMN = Bytes.toBytes("edge_f");
	
	//图像信息表名称
	private static final String imageTableName = "imageinfo";

	//文件系统对象，用于访问HDFS上的图像文件
	FileSystem fs = null;

	//支持HBase操作的map函数
	@Override
	protected void map(ImmutableBytesWritable rowKey, Result result,
			Context context) throws IOException, InterruptedException {
		//rowKey为HBase的列ID
		String key = new String(rowKey.get());
		//根据列ID打开图像文件
		FSDataInputStream file = fs.open(new Path("/imagesearch/images/" + key + ".jpg"));
		
		//将图像文件读取为图像对象
		BufferedImage image = ImageIO.read(file.getWrappedStream());
		
		//提取颜色直方图特征
		LireFeature colorFeature = new SimpleColorHistogram();
		colorFeature.extract(image);
		//提取边缘直方图特征
		LireFeature edgeFeature = new EdgeHistogram();
		edgeFeature.extract(image);
		
		//将颜色直方图特征和边缘直方图特征的二进制表示放入HBase表中。
		//Map任务结束后将会把数据插入到HBase中，特征提取任务完成。
		Put put = new Put(rowKey.get());
		put.add(COLUMN_FAMILY, COLOR_FEATURE_COLUMN, colorFeature.getByteArrayRepresentation());
		put.add(COLUMN_FAMILY, EDGE_FEATURE_COLUMN, edgeFeature.getByteArrayRepresentation());
		context.write(rowKey, put);
	}

	//此函数将在map任务执行之前执行，用于进行初始化操作
	@Override
	protected void setup(Context context) throws IOException,
			InterruptedException {
		//初始化文件系统对象，连接到Hadoop平台的HDFS上
		Configuration config = context.getConfiguration();
		fs = FileSystem.get(config);
	}

	/**
	 * Job configuration.
	 */
	public static Job configureJob(Configuration conf)
			throws IOException {

		TableMapReduceUtil.addDependencyJars(conf, ImageIndexJob.class, LireFeature.class);
		
		JobConf jobConf = new JobConf(conf);
		jobConf.setJobName("image-index");
		
		Job job = new Job(jobConf);
		job.setJarByClass(ImageIndexJob.class);
		
		Scan scan = new Scan();
		scan.setCaching(500);        // 1 is the default in Scan, which will be bad for MapReduce jobs
		scan.setCacheBlocks(false);  // don't set to true for MR jobs
		// set other scan attrs
		
		TableMapReduceUtil.initTableMapperJob(
				imageTableName,        // input table
				scan,               // Scan instance to control CF and attribute selection
				ImageIndexJob.class,     // mapper class
				null,         // mapper output key
				null,  // mapper output value
				job);
		
		TableMapReduceUtil.initTableReducerJob(
				imageTableName,      // output table
				null,             // reducer class
				job);
		job.setNumReduceTasks(0);
		
		return job;
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		
	    conf.set("fs.defaultFS", "hdfs://cluster1.centos:8020");
	    conf.set("yarn.resourcemanager.address", "cluster1.centos:8032");
	    conf.set("mapreduce.framework.name", "yarn");
	    
		Job job = configureJob(conf);

		int result = job.waitForCompletion(true) ? 0 : 1;
		System.exit(result);
	}
}
