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
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.TreeMap;

import javax.imageio.ImageIO;

import net.semanticmetadata.lire.imageanalysis.EdgeHistogram;
import net.semanticmetadata.lire.imageanalysis.LireFeature;
import net.semanticmetadata.lire.imageanalysis.SimpleColorHistogram;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.io.ImmutableBytesWritable;
import org.apache.hadoop.hbase.mapreduce.MultiTableOutputFormat;
import org.apache.hadoop.hbase.mapreduce.TableInputFormat;
import org.apache.hadoop.hbase.mapreduce.TableMapReduceUtil;
import org.apache.hadoop.hbase.protobuf.ProtobufUtil;
import org.apache.hadoop.hbase.protobuf.generated.ClientProtos;
import org.apache.hadoop.hbase.util.Base64;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.util.GenericOptionsParser;

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
public class IndexBuilder extends 
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

		ClientProtos.Scan proto = ProtobufUtil.toScan(new Scan());
		String scanString = Base64.encodeBytes(proto.toByteArray());

		JobConf jobConf = new JobConf(conf);
		jobConf.setJobName("image-index");
		
		jobConf.set(TableInputFormat.SCAN, scanString);
		jobConf.set(TableInputFormat.INPUT_TABLE, imageTableName);
		Job job = new Job(jobConf);
		job.setJarByClass(IndexBuilder.class);
		job.setMapperClass(IndexBuilder.class);
		job.setNumReduceTasks(0);
		job.setInputFormatClass(TableInputFormat.class);
		job.setOutputFormatClass(MultiTableOutputFormat.class);
		return job;
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = HBaseConfiguration.create();
		
	    conf.set("fs.defaultFS", "hdfs://cluster1.centos:8020");
	    conf.set("yarn.resourcemanager.address", "cluster1.centos:8032");
	    conf.set("mapreduce.framework.name", "yarn");
	    
		Job job = configureJob(conf);

		int result = job.waitForCompletion(true) ? 0 : 1;
		System.exit(result);
	}
}
