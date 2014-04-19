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
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.MultiTableOutputFormat;
import org.apache.hadoop.hbase.mapreduce.TableInputFormat;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;

public class ImageIndexJob extends 
	Mapper<ImmutableBytesWritable, Result, ImmutableBytesWritable, Put> {
	/** the column family containing the indexed row key */
	public static final byte[] COLUMN_FAMILY = Bytes.toBytes("i");
	public static final byte[] COLOR_FEATURE_COLUMN = Bytes.toBytes("color_f");
	public static final byte[] EDGE_FEATURE_COLUMN = Bytes.toBytes("edge_f");
	
	private static final String imageTableName = "imageinfo";
	
	private ImmutableBytesWritable tableName = 
			new ImmutableBytesWritable(Bytes.toBytes(imageTableName));

	FileSystem fs = null;

	@Override
	protected void map(ImmutableBytesWritable rowKey, Result result,
			Context context) throws IOException, InterruptedException {

		String key = new String(rowKey.get());
		FSDataInputStream file = fs.open(new Path("/imagesearch/images/" + key + ".jpg"));
		
		BufferedImage image = ImageIO.read(file.getWrappedStream());
		
		LireFeature colorFeature = new SimpleColorHistogram();
		colorFeature.extract(image);
		LireFeature edgeFeature = new EdgeHistogram();
		edgeFeature.extract(image);
		
		Put put = new Put(rowKey.get());
		put.add(COLUMN_FAMILY, COLOR_FEATURE_COLUMN, colorFeature.getByteArrayRepresentation());
		put.add(COLUMN_FAMILY, EDGE_FEATURE_COLUMN, edgeFeature.getByteArrayRepresentation());
		context.write(tableName, put);
	}

	@Override
	protected void setup(Context context) throws IOException,
			InterruptedException {
		Configuration config = context.getConfiguration();
		fs = FileSystem.get(config);
	}

	/**
	 * Job configuration.
	 */
	public static Job configureJob(Configuration conf)
			throws IOException {

		JobConf jobConf = new JobConf(conf);
		jobConf.setJobName("image-index");
		
		jobConf.set(TableInputFormat.INPUT_TABLE, imageTableName);
		Job job = new Job(jobConf);
		job.setJarByClass(ImageIndexJob.class);
		job.setMapperClass(ImageIndexJob.class);
		job.setNumReduceTasks(0);
		job.setInputFormatClass(TableInputFormat.class);
		job.setOutputFormatClass(MultiTableOutputFormat.class);
		return job;
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		
	    conf.set("fs.defaultFS", "hdfs://cluster1.centos:8020");
	    //conf.set("yarn.resourcemanager.address", "cluster1.centos:8032");
	    //conf.set("mapreduce.framework.name", "yarn");
	    
		Job job = configureJob(conf);

		int result = job.waitForCompletion(true) ? 0 : 1;
		System.exit(result);
	}
}
